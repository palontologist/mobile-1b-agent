/**
 * AgentLoop - The stable autonomy loop for FunctionGemma 270M on SM-A14.
 * 
 * Philosophy: Perceive → Reason (FunctionGemma) → Act (ActionRegistry) → Memory → Loop
 * Interval: 500ms (configurable)
 * Fallback: NPU → CPU graceful degrade (handled at LiteRT CompiledModel level)
 * 
 * Phase 2 Features:
 * - Calendar integration (Google Calendar API)
 * - SQLite-based note persistence
 * - TtsConfig for text-to-speech
 * - ErrorLogger for crash reporting
 * - BatteryOptimizer for power management
 * 
 * Key stability patterns:
 * - Batch size 1 for RAM safety (3.5G on SM-A14)
 * - Error boundaries per action (one failure ≠ loop crash)
 * - TTS only on user-initiated events (not spurious)
 * - Context window limited to last 5 interactions
 * - Battery-aware loop interval
 */
package com.ai.edge.agent

import android.content.Context
import android.os.Handler
import android.util.Log
import java.util.*
import kotlin.concurrent.*;

/**
 * The main agent perception-reason-action loop.
 * 
 * Runs as a Runnable posted to Main Looper at fixed intervals.
 * Each iteration:
 *   1. PERCEIVE: Gather context (audio/sensors/notifications/calendar)
 *   2. REASON:   FunctionGemma.generate(prompt) → JSON function call
 *   3. ACT:      ActionRegistry.execute(function, parameters, ctx) → ActionResult
 *   5. MEMORY:   Store (input, function, outcome) in SQLite
 *   6. LOOP:     Post self to handler with battery-aware interval
 */
class AgentLoop(
    private val ctx: Context,
    private val functionGemma: CompiledModel,  // LiteRT CompiledModel (NPU or CPU)
    private val inputContext: String = "User command context",  // From AudioRecord/STT
    private val loopIntervalMs: Long = 500
) : Runnable {

    // -----------------------------------------------------------------
    // Components (Phase 2 additions)
    // -----------------------------------------------------------------
    private val handler = Handler(ctx.mainLooper)
    private val batteryOptimizer = BatteryOptimizer().apply { init(ctx) }
    private val calendarService = CalendarService(ctx, "")
    private val noteManager = NoteManager().apply { init(ctx) }
    private val ttsConfig = TtsConfig().apply { init(ctx) }
    private val errorLogger = ErrorLogger()
    
    private val memory = NoteDatabase(ctx)  // SQLite-based persistence
    private var consecutiveErrors = 0
    private val maxConsecutiveErrors = 3

    // -----------------------------------------------------------------
    // Agent Loop Entry Point
    // -----------------------------------------------------------------

    /** Start the perpetual agent loop. */
    fun start() {
        handler.post(this)
        Log.i("AgentLoop", "Agent loop started — ${loopIntervalMs}ms intervals")
    }

    /** Stop the loop (e.g., on app termination). */
    fun stop() {
        handler.removeCallbacks(this)
        Log.i("AgentLoop", "Agent loop stopped")
        ttsConfig.shutdown()
    }

    // -----------------------------------------------------------------
    // Core Loop Implementation (Runnable)
    // -----------------------------------------------------------------

    override fun run() {
        try {
            // ---- 1. PERCEIVE ----
            // Gather context from various sources
            val perceivedContext = perceiveContext()

            // ---- 2. REASON ----
            // Build prompt for FunctionGemma including memory context
            val prompt = buildPrompt(perceivedContext)

            // Run FunctionGemma inference
            val gemmaOutput = runFunctionGemma(prompt)

            // Parse the JSON function call from model output
            val parsed = parseFunctionCall(gemmaOutput)
            if (parsed == null) {
                // Model didn't output a recognized function → no-op
                Log.w("AgentLoop", "FunctionGemma output had no recognized function: $gemmaOutput")
                consecutiveErrors = 0  // Not an error, just no action
            } else {
                val functionName = parsed.function
                val parameters = parsed.parameters ?: emptyMap()

                // ---- 3. ACT ----
                val result = ActionRegistry.execute(
                    functionName,
                    parameters,
                    ctx
                )

                // ---- 4. MEMORY ----
                // Persist the full cycle in SQLite for context continuity
                memory.store(
                    input = perceivedContext,
                    function = functionName,
                    outcome = result.success ? "success" : "error: ${result.error}",
                    speakText = result.speak_text,
                    shouldSpeak = result.should_speak
                )

                // Update error tracking
                if (result.success) {
                    consecutiveErrors = 0  // Reset on success
                } else {
                    consecutiveErrors++
                    errorLogger.logError(
                        ctx,
                        "AgentLoop",
                        "Action $functionName failed (consecutive: $consecutiveErrors/${maxConsecutiveErrors})"
                    )
                }

                // Optionally speak result (TTS) - but only for notable outcomes
                TtsConfig.speakIfNeeded(result.should_speak, result.speak_text, ctx)
            }

        } catch (e: Exception) {
            // ---- Error Boundary ----
            consecutiveErrors++
            errorLogger.logError(
                ctx,
                "AgentLoop",
                "Uncaught error in agent loop iteration $consecutiveErrors",
                e
            )

            // Longer backoff after multiple errors
            if (consecutiveErrors >= maxConsecutiveErrors) {
                // 2s pause instead of standard interval
                handler.postDelayed(this, 2000)
                consecutiveErrors = 0  // Reset after backoff
                return
            }
        }

        // ---- 5. Loop (reschedule with battery awareness) ----
        // Get battery-aware interval and reschedule
        val nextInterval = batteryOptimizer.getRecommendedInterval()
        if (!Thread.currentThread().isInterrupted) {
            handler.postDelayed(this, nextInterval)
        }
    }

    // -----------------------------------------------------------------
    // 1. PERCEIVE: Gather context from various sources
    // -----------------------------------------------------------------
    private fun perceiveContext(): String {
        // 1. Start with the input context (from last STT result)
        var context = inputContext
        
        // 2. Add calendar events (next 1-2 events)
        val calendarEvents = calendarService.getTodayEvents()
        if (calendarEvents.isNotEmpty()) {
            context = "$context | Calendar: ${calendarEvents.joinToString(" | ")}"
        }
        
        // 3. Add recent notes (last 2)
        val recentNotes = noteManager.getRecentNotes(2)
        if (recentNotes.isNotEmpty()) {
            val noteText = recentNotes.joinToString(" | ")
            context = "$context | Notes: $noteText"
        }
        
        return context
    }

    // -----------------------------------------------------------------
    // 2. REASON: Build prompt for FunctionGemma
    // -----------------------------------------------------------------
    private fun buildPrompt(currentContext: String): String {
        // Retrieve last 5 interactions from SQLite for context
        val recent interactions: List<String> = memory.getRecentInteractions(5)

        val contextSection = if (recent interactions.isNotEmpty()) {
            "Previous interactions:\n" +
                recent interactions.joinToString("\n") + "\n\n"
        } else {
            ""
        }

        val basePrompt = """
            |You are FunctionGemma 270M on-device AI for Samsung SM-A14.
            |Available actions (return JSON only, no prose):
            
            |open_flashlight       → {"name": "open_flashlight", "parameters": {}}
            |close_flashlight     → {"name": "close_flashlight", "parameters": {}}
            |query_calendar       → {"name": "query_calendar", "parameters": {}}
            |append_note(title: str, content: str) → {"name": "append_note", "parameters": {"title": "...", "content": "..."}}
            |set_alarm(hh:mm)     → {"name": "set_alarm", "parameters": {"time": "..."}}
            |take_photo           → {"name": "take_photo", "parameters": {}}
            |send_message(contact, message) → {"name": "send_message", "parameters": {"contact": "...", "message": "..."}}
            |noop                 → {"name": "noop", "parameters": {}}
            |
            |User context: "$currentContext"
            |$contextSection
            |Respond with ONLY the JSON object for the desired action.
            |Example: {"name": "open_flashlight", "parameters": {}}
            |Do not add any reasoning, apologies, or additional text.
            |""".trimMargin()

        return basePrompt
    }

    // -----------------------------------------------------------------
    // 3. ACT: Execute action via ActionRegistry
    // -----------------------------------------------------------------
    // (Same as before - ActionRegistry.execute)

    // -----------------------------------------------------------------
    // 4. MEMORY: SQLite-based persistence
    // -----------------------------------------------------------------
    // (Already in AgentMemory → replaced with NoteDatabase)

    // -----------------------------------------------------------------
    // 5. Loop with battery awareness
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // FunctionGemma Inference (LiteRT CompiledModel)
    // -----------------------------------------------------------------
    private fun runFunctionGemma(prompt: String): String {
        try {
            // Prepare input buffer - batch size 1 for 3.5G RAM safety
            val inputBuffers = functionGemma.createInputBuffers()
            val outputBuffers = functionGemma.createOutputBuffers()

            // TODO: Proper tokenization/embedding of prompt string
            // For now, placeholder: the CompiledModel expects float input
            
            // For now, simulate a function call based on prompt keywords
            return simulateFunctionGemmaResponse(prompt)

        } catch (e: Exception) {
            Log.e("AgentLoop", "FunctionGemma inference error: ${e.message}")
            return "{}"
        }
    }

    // -----------------------------------------------------------------
    // Simulated FunctionGemma response (placeholder)
    // -----------------------------------------------------------------
    private fun simulateFunctionGemmaResponse(prompt: String): String {
        val lower = prompt.toLowerCase()

        // Keyword-based function selection (placeholder)
        if (lower.contains("flash")) {
            return """{"name": "open_flashlight", "parameters": {}}"""
        } else if (lower.contains("calendar") || lower.contains("event")) {
            return """{"name": "query_calendar", "parameters": {}}"""
        } else if (lower.contains("note")) {
            return """{"name": "append_note", "parameters": {"title": "Test", "content": "Test note from agent loop"}}"""
        } else if (lower.contains("alarm")) {
            return """{"name": "set_alarm", "parameters": {"time": "09:00"}}"""
        } else if (lower.contains("photo") || lower.contains("picture")) {
            return """{"name": "take_photo", "parameters": {}}"""
        } else {
            return """{"name": "noop", "parameters": {}}"""
        }
    }

    // -----------------------------------------------------------------
    // Function Call JSON Parsing
    // -----------------------------------------------------------------
    private data class ParsedFunction(
        val function: String,
        val parameters: Map<String, Any>? = null
    )

    private fun parseFunctionCall(gemmaOutput: String): ParsedFunction? {
        try {
            val cleaned = gemmaOutput.trim().replace("\n", " ").trim()
            val jsonStart = cleaned.indexOf('{')
            val jsonEnd = cleaned.lastIndexOf('}') + 1
            if (jsonStart < 0 || jsonEnd <= jsonStart) return null

            val jsonStr = cleaned.substring(jsonStart, jsonEnd)
            val parse = JSONObject(jsonStr)

            val functionName = parse.getString("name")
            val parameters = if (parse.has("parameters")) {
                val paramsMap = mutableMapOf<String, Any>()
                val opt = parse.getJSONObject("parameters")
                val keys = opt.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    paramsMap[key] = when (opt.get(key)) {
                        is JSONObject -> opt.getJSONObject(key).toString()
                        is JSONArray -> opt.getJSONArray(key).toString()
                        else -> opt.getString(key)
                    }
                }
                paramsMap.toMutableMap()
            } else {
                null
            }

            if (ActionRegistry.isFunctionRegistered(functionName)) {
                return ParsedFunction(functionName, parameters)
            } else {
                Log.w("AgentLoop", "Unregistered function from model: $functionName")
                return null
            }
        } catch (e: Exception) {
            Log.e("AgentLoop", "Failed to parse FunctionGemma output: ${e.message}", e)
            return null
        }
    }

    // -----------------------------------------------------------------
    // TTS Helper (using TtsConfig)
    // -----------------------------------------------------------------
    private fun speak(text: String) {
        TtsConfig.speak(text, ctx)
    }

    // -----------------------------------------------------------------
    // Inner Classes (Phase 2 components)
    // -----------------------------------------------------------------

    /** BatteryOptimizer - Phase 2 component */
    private class BatteryOptimizer(private var ctx: Context?) {
        private var lastCheck = 0L
        private val CHECK_INTERVAL = 60000L

        fun init(c: Context) = context = c

        fun getRecommendedInterval(): Long {
            if (context == null) return 500L

            val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            val level = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            val isCharging = batteryManager.isChargingPlugPresent(BatteryManager.PLUG_AC)
            val batteryTemperature = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_TEMPERATURE) / 10

            var interval = 500L
            when {
                isChargingPlugPresent(BatteryManager.PLUG_AC) -> interval = 500L
                level <= 15 -> interval = 5000L
                level <= 30 -> interval = 1000L
                batteryTemperature >= 45 -> interval = 2000L
                else -> interval = 500L
            }

            val now = System.currentTimeMillis()
            if (now - lastCheck < 60000L) return interval
            lastCheck = now

            Log.d("BatteryOptimizer", "Battery: ${level}%, Interval: ${interval}ms")
            return interval
        }

        private fun isChargingPlugPresent(plug: String): Boolean {
            // Simplified check - in production use the full API
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            return bm.isCharging
        }
    }

    /** NoteManager - Phase 2 component (SQLite-based) */
    private class NoteManager(private val ctx: Context) {
        private lateinit var dbHelper: NoteDbHelper

        init {
            dbHelper = NoteDbHelper(ctx)
        }

        fun appendNote(title: String, content: String): Boolean {
            return NoteManagerKt.appendNote(title, content)
        }

        fun getRecentNotes(count: Int): List<NoteDbHelper.Note> {
            return dbHelper.getRecentNotes(count)
        }
    }

    /** ErrorLogger - Phase 2 component */
    private class ErrorLoggerImpl {
        fun logError(context: Context, tag: String, message: String, throwable: Throwable?) {
            ErrorLogger.logError(context, tag, message, throwable)
        }
    }
}

/**
 * Helper object for NoteManagerKt (SQLite operations)
 */
object NoteManagerKt {
    private lateinit var dbHelper: NoteDbHelper

    fun init(context: Context) {
        dbHelper = NoteDbHelper(context)
    }

    fun appendNote(title: String, content: String): Boolean {
        return dbHelper.insertNote(Note(
            title = title,
            content = content,
            modified = System.currentTimeMillis()
        ))
    }

    /** Get last N notes */
    fun getRecentNotes(count: Int): List<NoteDbHelper.Note> {
        return dbHelper.getRecentNotes(count)
    }
}