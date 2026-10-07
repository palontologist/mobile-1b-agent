package com.ai.edge.agent

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Gemma 4 E2B benchmark on real hardware.
 *
 * The question this exists to answer: is a small LLM viable as the reasoning tier
 * of the local brain on a budget phone, or is the retrieval-only assistant the
 * honest ceiling? Everything so far is either measured on this device (the MiniLM
 * router) or extrapolated from other hardware (every vendor table). This measures
 * it here.
 *
 * What it records, in the order that matters:
 *   - engine load time, because a 20 s first answer is not usable regardless of
 *     tokens per second;
 *   - time to first token and decode rate, from the streaming callback;
 *   - peak RSS at each stage, from VmHWM so it survives a GC mid-run;
 *   - tool-call accuracy, which is the only number that decides whether the model
 *     is useful at all here.
 *
 * Output is a single greppable stream on tag [GemmaBench] so it can be scraped
 * from logcat with no UI interaction:
 *
 *   adb shell am start -n com.ai.edge.agent/.GemmaBenchActivity \
 *       --es model /sdcard/gemma-4-E2B-it.litertlm
 */
class GemmaBenchActivity : Activity() {

  private lateinit var log: TextView
  private val lines = StringBuilder()

  /** Rough only: Gemma tokenises English at roughly 4 characters per token. */
  private fun estTokens(s: String): Int = (s.length / 4).coerceAtLeast(1)

  private fun peakRssMb(): Long =
      try {
        val status =
            File("/proc/${android.os.Process.myPid()}/status").readText()
        val kb =
            status
                .lineSequence()
                .first { it.startsWith("VmHWM:") }
                .split(Regex("\\s+"))[1]
                .toLong()
        kb / 1024
      } catch (e: Exception) {
        -1
      }

  private fun say(s: String) {
    Log.i(TAG, s)
    lines.append(s).append('\n')
    runOnUiThread { log.text = lines.toString() }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    log = TextView(this)
    log.textSize = 9f
    setContentView(log)

    val modelPath = intent.getStringExtra("model") ?: DEFAULT_MODEL
    Thread { run(modelPath) }.start()
  }

  private fun run(modelPath: String) {
    say(
        "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT} " +
            "ram_mb=${totalRamMb()} rss_before_mb=${peakRssMb()}"
    )

    val f = File(modelPath)
    if (!f.exists()) {
      say("FAIL model not found at $modelPath")
      return
    }
    say("MODEL file_mb=${f.length() / (1024 * 1024)}")

    // --- engine load -------------------------------------------------------
    val t0 = System.currentTimeMillis()
    var engine: Engine? = null
    try {
      engine =
          Engine(
              EngineConfig(
                  modelPath = modelPath,
                  backend = Backend.CPU(threadCount = 4),
                  cacheDir = File(cacheDir, "litertlm").absolutePath,
              )
          )
      engine.initialize()
    } catch (t: Throwable) {
      say("FAIL engine init after ${System.currentTimeMillis() - t0}ms: $t")
      return
    }
    val loadMs = System.currentTimeMillis() - t0
    say("ENGINE_INIT ms=$loadMs rss_mb=${peakRssMb()}")

    // --- generation --------------------------------------------------------
    val gen = engine.createConversation(ConversationConfig())
    val prompts =
        listOf(
            "Name three rivers in Europe, one sentence.",
            "Explain in one sentence why the sky looks red at sunset.",
            "Write a two sentence note to a neighbour about a package.",
        )

    val ttfts = mutableListOf<Long>()
    val rates = mutableListOf<Double>()
    for (p in prompts) {
      val r = stream(gen, p, maxTokens = 128) ?: continue
      ttfts += r.firstChunkMs
      val decodeSec = (r.totalMs - r.firstChunkMs) / 1000.0
      if (decodeSec > 0 && r.chars > 0) rates += (estTokens(r.text) / decodeSec)
      say(
          "GEN chars=${r.chars} est_tokens=${estTokens(r.text)} " +
              "ttft_ms=${r.firstChunkMs} total_ms=${r.totalMs} " +
              "out=${r.text.replace('\n', ' ').take(90)}"
      )
    }
    say("GEN_SUMMARY ttft_median_ms=${ttfts.medianOr0()} decode_toks_per_sec=${rates.averageOrNull()}")

    // --- tool-call accuracy ------------------------------------------------
    val tools = assistantLocalAssistantTools()
    var correct = 0
    var noOutput = 0
    val perTool = mutableMapOf<String, Pair<Int, Int>>() // tool -> (hit, total)

    for ((prompt, expected) in toolCases()) {
      // A fresh conversation per case, and this is not a style choice. Reusing one
      // conversation accumulates 15 turns of tool schemas, prompts and replies into
      // a 1024-entry KV cache. Cases after the sixth die with "Prefill input length
      // exceeds available state entries", and every one of them scores as a miss --
      // so the run measures context exhaustion and reports it as bad tool selection.
      // Each case here is independent, so each gets an empty context.
      val convo = engine.createConversation(ConversationConfig())
      val full = "$tools\n\nUser request: $prompt\n\nAnswer with only a JSON object of the form {\"tool\":\"<name>\",\"arguments\":{}}. No other text."
      val r = stream(convo, full, maxTokens = 64)
      if (r?.text.isNullOrBlank()) noOutput++
      val got = extractToolName(r?.text ?: "")
      val hit = got == expected
      if (hit) correct++
      val cur = perTool[expected] ?: (0 to 0)
      perTool[expected] = (cur.first + (if (hit) 1 else 0)) to (cur.second + 1)
      say("TOOL hit=$hit expected=$expected got=${got ?: "null"} ms=${r?.totalMs} prompt=${prompt.take(48)}")
    }

    say("TOOL_SUMMARY correct=$correct total=${toolCases().size} accuracy=${correct.toDouble() / toolCases().size} no_output=$noOutput")
    for ((t, v) in perTool.entries.sortedBy { it.key }) {
      say("TOOL_PER_TOOL $t ${v.first}/${v.second}")
    }

    say("FINAL rss_mb=${peakRssMb()}")
    engine.close()
    say("DONE")
  }

  private data class Gen(val text: String, val chars: Int, val firstChunkMs: Long, val totalMs: Long)

  /**
   * Streams one turn, recording when the first chunk arrived.
   *
   * Blocking sendMessage() could give total time but not time-to-first-token, and
   * TTFT is the number that decides whether the model feels responsive. A 600 ms
   * router followed by 4 s of silence is a bad experience even if decode is quick.
   */
  private fun stream(c: Conversation, prompt: String, maxTokens: Int): Gen? {
    val start = System.currentTimeMillis()
    val sb = StringBuilder()
    var first = -1L
    val done = AtomicBoolean(false)
    // Explicitly typed: bare nulls make the compiler unable to choose between the
    // MessageCallback and Flow overloads of sendMessageAsync.
    val noExtras: Map<String, Any> = emptyMap()
    val noRepetition: com.google.ai.edge.litertlm.RepetitionPenaltyConfig? = null
    val noNgram: com.google.ai.edge.litertlm.NoRepeatNgramConfig? = null
    val noSuppress: com.google.ai.edge.litertlm.SuppressTokensConfig? = null
    val noThinking: com.google.ai.edge.litertlm.ThinkingConfig? = null
    val noFormat: com.google.ai.edge.litertlm.ResponseFormat? = null

    return try {
      c.sendMessageAsync(
          prompt,
          object : MessageCallback {
            override fun onMessage(m: Message) {
              if (first < 0) first = System.currentTimeMillis() - start
              sb.append(m.toString())
            }

            override fun onDone() {
              done.set(true)
            }

            override fun onError(t: Throwable) {
              Log.w(TAG, "stream error", t)
              done.set(true)
            }
          },
          noExtras,
          noRepetition,
          noNgram,
          noSuppress,
          maxTokens,
          noThinking,
          noFormat,
      )
      // The callback is dispatched on the engine's own threads; wait it out.
      val deadline = start + 300_000
      while (!done.get() && System.currentTimeMillis() < deadline) Thread.sleep(50)
      val total = System.currentTimeMillis() - start
      if (first < 0) {
        say("STREAM empty after ${total}ms")
        null
      } else {
        Gen(sb.toString(), sb.length, first, total)
      }
    } catch (t: Throwable) {
      say("STREAM_FAIL ${t}")
      null
    }
  }

  /**
   * Tools the assistant would actually have.
   *
   * Only offline-capable tools appear. Including weather and web search would
   * inflate the score with calls the assistant cannot serve without a network, and
   * the point of this run is to measure the offline case.
   */
  private fun assistantLocalAssistantTools(): String =
      """
      You can call exactly one of these tools:
      get_current_time() - the current date and time
      get_location() - the device's GPS coordinates
      search_conversations(query, speaker) - search recorded conversations
      list_tasks() - tasks and reminders stored on the device
      add_task(title) - create a task
      add_reminder(title, when) - create a reminder
      set_timer(minutes) - set a countdown
      """.trimIndent()

  private fun toolCases(): List<Pair<String, String>> =
      listOf(
          "what time is it right now" to "get_current_time",
          "what is today's date" to "get_current_time",
          "where am I" to "get_location",
          "what are my coordinates" to "get_location",
          "what did marco say about bread" to "search_conversations",
          "search my conversations for anything about invoices" to "search_conversations",
          "what did the other person say about the invoice" to "search_conversations",
          "what is on my list" to "list_tasks",
          "show my tasks" to "list_tasks",
          "what reminders do I have" to "list_tasks",
          "add a task to pay the electric bill" to "add_task",
          "remind me to call the dentist" to "add_reminder",
          "add a task: buy milk" to "add_task",
          "set a timer for 10 minutes" to "set_timer",
          "start a 5 minute countdown" to "set_timer",
      )

  /** Pulls the tool name out of the reply, tolerating prose around the JSON. */
  private fun extractToolName(s: String): String? {
      val m = Regex("\"tool\"\\s*:\\s*\"([a-z_]+)\"").find(s)
      return m?.groupValues?.get(1)
  }

  private fun totalRamMb(): Long =
      try {
        val am = getSystemService(android.content.Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val mi = android.app.ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        mi.totalMem / (1024 * 1024)
      } catch (e: Exception) {
        -1
      }

  private fun List<Long>.medianOr0(): Long =
      if (isEmpty()) -1 else sorted()[size / 2]

  private fun List<Double>.averageOrNull(): String =
      if (isEmpty()) "n/a" else String.format("%.2f", average())

  companion object {
    const val TAG = "GemmaBench"
    const val DEFAULT_MODEL = "/sdcard/gemma-4-E2B-it.litertlm"
  }
}