/**
 * AgentLoop - The stable autonomy loop for FunctionGemma 270M on SM-A14.
 * 
 * Philosophy: Perceive → Reason (FunctionGemma) → Act (ActionRegistry) → Memory → Loop
 * Interval: 500ms (configurable)
 * Fallback: NPU → CPU graceful degrade (handled at LiteRT CompiledModel level)
 * 
 * Key stability patterns:
 * - Batch size 1 for RAM safety (3.5G on SM-A14)
 * - Error boundaries per action (one failure ≠ loop crash)
 * - TTS only on user-initiated events (not spurious)
 * - Context window limited to last 5 interactions
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
 *   1. Perceive: Gather context (audio/sensors/notifications - simplified here)
 *   2. Reason:   FunctionGemma.generate(prompt) → JSON function call
 *   3. Act:      ActionRegistry.execute(function, parameters, ctx)
 *   4. Memory:   Store (input, function, outcome) in Room/SQLite
 *   5. Loop:     Post self to handler again after interval
 */
class AgentLoop(
    private val ctx: Context,
    private val functionGemma: CompiledModel,  // LiteRT CompiledModel (NPU or CPU)
    private val inputContext: String = "User command context",  // From AudioRecord/STT
    private val loopIntervalMs: Long = 500
) : Runnable {

    private val handler = Handler(ctx.mainLooper)
    private val memory = AgentMemory(ctx)  // Simple SQLite/Room wrapper
    private var consecutiveErrors = 0
    private val maxConsecutiveErrors = 3

    // -----------------------------------------------------------------
    // Agent Loop Entry Point
    // -----------------------------------------------------------------

    /**
     * Start the perpetual agent loop.
     * Call once on app startup (after FunctionGemma model loaded).
     */
    fun start() {
        handler.post(this)
        Log.i("AgentLoop", "Agent loop started — ${loopIntervalMs}ms intervals")
    }

    /** Stop the loop (e.g., on app termination). */
    fun stop() {
        handler.removeCallbacks(this)
        Log.i("AgentLoop", "Agent loop stopped")
    }

    // -----------------------------------------------------------------
    // Core Loop Implementation (Runnable)
    // -----------------------------------------------------------------

    override fun run() {
        try {
            // ---- 1. PERCEIVE ----
            // In production: AudioRecord → STT → extract user intent / context
            // Here we use the pre-supplied inputContext (from last STT result)
            val perceivedContext = inputContext

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
                // Persist the full cycle for context continuity
                memory.store(
                    input = perceivedContext,
                    function = functionName,
                    outcome = result.success ? "success" : "error: ${result.error}",
                    confidence = 1.0f,  // FunctionGemma confidence would go here
                    speakText = result.speak_text,
                    shouldSpeak = result.should_speak
                )

                // Update error tracking
                if (result.success) {
                    consecutiveErrors = 0  // Reset on success
                } else {
                    consecutiveErrors++
                    Log.w(
                        "AgentLoop",
                        "Action $functionName failed (consecutive: $consecutiveErrors/${maxConsecutiveErrors})"
                    )
                }

                // Optionally speak result (TTS) - but only for notable outcomes
                if (result.should_speak && result.speak_text != null) {
                    speak(result.speak_text)
                }
            }

        } catch (e: Exception) {
            // ---- Error Boundary ----
            consecutiveErrors++
            Log.e("AgentLoop", "Uncaught error in agent loop iteration $consecutiveErrors", e)

            // Longer backoff after multiple errors
            if (consecutiveErrors >= maxConsecutiveErrors) {
                // 2s pause instead of 500ms to avoid tight error loop
                handler.postDelayed(this, 2000)
                consecutiveErrors = 0  // Reset after backoff
                return
            }
        }

        // ---- 5. Loop (reschedule) ----
        // Only post if not stopped
        if (!Thread.currentThread().isInterrupted) {
            handler.postDelayed(this, loopIntervalMs)
        }
    }

    // -----------------------------------------------------------------
    // Prompt Building: Include memory/context for FunctionGemma
    // -----------------------------------------------------------------
    private fun buildPrompt(currentContext: String): String {
        // Retrieve last 3 interactions from memory to include as context
        val recent = memory.getRecentContexts(3)

        val contextSection = if (recent.isNotEmpty()) {
            "Previous interactions:\n" +
                recent.joinToString("\n") + "\n\n"
        } else {
            ""
        }

        val basePrompt = """
            |You are FunctionGemma 270M on-device AI for Samsung SM-A14.
            |Available actions (return JSON only, no prose):
            |
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
    // FunctionGemma Inference (LiteRT CompiledModel)
    // -----------------------------------------------------------------
    private fun runFunctionGemma(prompt: String): String {
        try {
            // Prepare input buffer - batch size 1 for 3.5G RAM safety
            val inputBuffers = functionGemma.createInputBuffers()
            val outputBuffers = functionGemma.createOutputBuffers()

            // TODO: Proper tokenization/embedding of prompt string
            // For now, placeholder: the CompiledModel expects float input
            // In production, use Tokenizer → input_ids → input_tensor
            
            // Simplified: feed prompt as float array (real impl needs proper tokenizer)
            // This is where the 270M model actually runs - tokenized text → logits → function probs
            
            // Placeholder implementation - in production this would be:
            // 1. tokenizer.encode(prompt) → input_ids
            // 2. input_ids → float32 tensor [1, seq_len, embed_dim]
            // 3. functionGemma.run(inputBuffers, outputBuffers)
            // 4. outputBuffers[0].readFloat() → decode function name

            // For now, simulate a function call based on prompt keywords
            return simulateFunctionGemmaResponse(prompt)

        } catch (e: Exception) {
            Log.e("AgentLoop", "FunctionGemma inference error: ${e.message}")
            return "{}"
        }
    }

    /**
     * Simulated FunctionGemma response for demo/prototyping.
     * In production, replace with actual CompiledModel inference.
     */
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
            // Default: no-op or ask for clarification
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
            // The model should output clean JSON like:
            // {"name": "open_flashlight", "parameters": {}}
            
            // Clean up any surrounding text (model may add newlines, prefixes)
            val cleaned = gemmaOutput.trim().replace("\n", " ").trim()
            
            // Extract JSON object (simple approach - find { … })
            val jsonStart = cleaned.indexOf('{')
            val jsonEnd = cleaned.lastIndexOf('}') + 1
            if (jsonStart < 0 || jsonEnd <= jsonStart) {
                return null
            }
            
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
                        is JSONObject -> opt.getJSONObject(key).toString() // nested if needed
                        is JSONArray -> opt.getJSONArray(key).toString()
                        else -> opt.getString(key)
                    }
                }
                paramsMap.toMutableMap()
            } else {
                null
            }
            
            // Validate function is registered
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
    // TTS Helper (lightweight, only on notable outcomes)
    // -----------------------------------------------------------------
    private fun speak(text: String) {
        try {
            val tts = ctx.getSystemService(TextToSpeech::class.java)
            if (tts.isLanguageAvailable("en-US") >= TextToSpeech.AVAILABLE) {
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "uniqueId")
            }
        } catch (e: Exception) {
            // TTS failure is non-critical - just log
            Log.d("AgentLoop", "TTS error (non-critical): ${e.message}")
        }
    }
}

/**
 * SimpleAgentMemory - Persists agent cycle context across app restarts.
 * 
 * In production, this would be a Room/SQLite database.
 * For this prototype, we use SharedPreferences with limited history.
 */
private class AgentMemory(
    private val ctx: Context
) {
    private val prefs = ctx.getSharedPreferences("agent_memory", Context.MODE_PRIVATE)
    private val GOTCHA_KEY = "agent_history"
    private val MAX_HISTORY = 10  // Keep last 10 interactions

    /**
     * Store one agent cycle: input, function, outcome, speak flags.
     */
    fun store(
        input: String,
        function: String,
        outcome: String,
        confidence: Float,
        speakText: String? = null,
        shouldSpeak: Boolean = false
    ) {
        try {
            val json = JSONObject().apply {
                put("input", input)
                put("function", function)
                put("outcome", outcome)
                put("timestamp", System.currentTimeMillis())
                put("confidence", confidence)
            }

            // Prepend to history (newest first)
            val history = getHistory()
            history.add(0, json.toString())

            // Trim to max
            val trimmed = history.take(MAX_HISTORY)

            // Save back
            val editor = prefs.edit()
            editor.putString(GOTCHA_KEY, trimmed.joinToString("\n"))
            editor.apply()
        } catch (e: Exception) {
            Log.d("AgentMemory", "Failed to store memory: ${e.message}")
        }
    }

    /**
     * Retrieve last N interaction contexts (newest first).
     */
    fun getRecentContexts(n: Int): List<String> {
        val history = getHistory()
        return history
            .subList(0, Math.min(n, history.size))
            .map { JSONObject(it) }
            .map { it.getString("function") + ": " + it.getString("outcome") }
    }

    private fun getHistory(): List<String> {
        val stored = prefs.getString(GOTCHA_KEY, "")
        if (stored.isBlank()) return emptyList()
        return stored.split("\n").filter { it.isNotBlank() }
    }
}