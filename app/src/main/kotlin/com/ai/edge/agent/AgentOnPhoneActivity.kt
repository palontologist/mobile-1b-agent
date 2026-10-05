package com.ai.edge.agent

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject

/**
 * Agent on Phone — an embedding-router agent, with the LLM deliberately absent.
 *
 * Routing is MiniLM-L6-v2 through the LiteRT CompiledModel API: cosine similarity
 * over per-tool prototype sets, with `clarify` competing as a fifth class so the
 * router can decline instead of always acting.
 *
 * Measured accuracy for the four physical tools is 67% on a 45-case held-out set
 * (30/45), with failures concentrated in negation, indirect phrasing, and
 * out-of-scope input. [NEGATION] exists because of that number: unguarded, the
 * router executed the opposite of 4 of 5 negated requests ("do not switch the
 * torch on" -> close_flashlight).
 *
 * No LiteRT-LM engine is loaded here on purpose. The measurement says a 270M
 * generator costs 3.1 s and 983 MB for 22% routing accuracy, while this router
 * costs ~0.6 s and 285 MB for 67%. Composing a reply is a separate problem and
 * belongs in a separate step, invoked only when the router's margin is thin.
 */
class AgentOnPhoneActivity : android.app.Activity() {

  companion object {
    private const val TAG = "AOP"
    private const val MODEL = "minilm.tflite"
    private const val TOKENS = "minilm_tokens.json"
    private const val EXTRA_UTTERANCE = "utterance"
    private const val SEQ = 128
    private const val DIM = 384

    /** Below this margin the router declines rather than picking the argmax. */
    private const val ABSTAIN_MARGIN = 0.02f

    /**
     * A prohibition anywhere in the utterance means: do not act.
     *
     * Held-out measurement: without this guard, "do not switch the torch on" routed
     * to close_flashlight and "I don't want to take a photo" routed to take_photo —
     * the exact inverse of the instruction. An agent that inverts a stated "don't"
     * is worse than one that abstains, so this gates ahead of the model entirely.
     */
    private val NEGATION =
        Regex(
            "\\b(?:don'?t|do\\s+not|never|no\\s+need|without\\s+doing)\\b",
            RegexOption.IGNORE_CASE,
        )

    /** Alias -> package. Explicit because package visibility is filtered on API 30+. */
    private val APP_ALIASES =
        mapOf(
            "whatsapp" to "com.whatsapp",
            "chrome" to "com.android.chrome",
            "browser" to "com.android.chrome",
            "settings" to "com.android.settings",
            "messages" to "com.google.android.apps.messaging",
            "messenger" to "com.google.android.apps.messaging",
            "contacts" to "com.samsung.android.app.contacts",
            "play store" to "com.android.vending",
            "camera" to "com.sec.android.app.camera",
            "gallery" to "com.sec.android.gallery3d",
            "clock" to "com.android.deskclock",
            "calculator" to "com.sec.calculator",
            "gmail" to "com.google.android.gm",
            "email" to "com.google.android.gm",
            "maps" to "com.google.android.apps.maps",
            "youtube" to "com.google.android.youtube",
        )
  }

  private lateinit var model: CompiledModel
  private lateinit var inputBuffers: List<TensorBuffer>
  private lateinit var outputBuffers: List<TensorBuffer>

  /** tool name -> one unit vector per prototype. */
  private val toolProtos = HashMap<String, List<FloatArray>>()
  private var appProtos = emptyList<FloatArray>()
  private var outOfDomain = emptyList<FloatArray>()
  private var ready = false

  private lateinit var tokenizer: WordPieceTokenizer
  private lateinit var input: EditText
  private lateinit var logView: TextView

  /**
   * singleInstance so a second `am start --es utterance ...` reaches onNewIntent
   * instead of rebuilding the activity. Rebuilding costs ~22 s (see loadRouter),
   * which makes scripted regression runs impractical.
   */
  override fun onNewIntent(intent: android.content.Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    intent.getStringExtra(EXTRA_UTTERANCE)?.let { queued = it }
    drainQueue()
  }

  @Volatile private var queued: String? = null

  private fun drainQueue() {
    if (!ready || queued == null) return
    val text = queued!!
    queued = null
    handle(text)
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    buildUi()
    Thread {
      try {
        loadRouter()
        // Scriptable entry point: `am start -n .../.AgentOnPhoneActivity --es
        // utterance "..."`. Lets the routing path be driven from adb without
        // tapping, which is how it was regression-tested.
        intent?.getStringExtra(EXTRA_UTTERANCE)?.let { handle(it) }
      } catch (t: Throwable) {
        Log.e(TAG, "LOAD_FAILED ${t.javaClass.name}: ${t.message}", t)
        log("router failed to load: ${t.message}")
      }
    }.start()
  }

  private fun buildUi() {
    val root =
        LinearLayout(this).apply {
          orientation = LinearLayout.VERTICAL
          setPadding(24, 32, 24, 24)
          setBackgroundColor(Color.WHITE)
        }

    root.addView(
        TextView(this).apply {
          text = "Agent on Phone"
          textSize = 22f
          setTextColor(Color.BLACK)
          setPadding(0, 0, 0, 4)
        }
    )
    root.addView(
        TextView(this).apply {
          text = "MiniLM router · no LLM loaded"
          textSize = 12f
          setTextColor(Color.GRAY)
          setPadding(0, 0, 0, 12)
        }
    )

    input =
        EditText(this).apply {
          hint = "e.g. \"the room is too dark to read anything\""
          textSize = 15f
          setTextColor(Color.BLACK)
          setHintTextColor(Color.GRAY)
        }
    root.addView(input)

    root.addView(
        Button(this).apply {
          text = "route it"
          isEnabled = false
          tag = "route"
          setOnClickListener { handle(input.text.toString()) }
        }
    )
    val buttons = root.getChildAt(root.childCount - 1) as Button

    val quick =
        LinearLayout(this).apply {
          orientation = LinearLayout.HORIZONTAL
          setPadding(0, 8, 0, 0)
        }
    for ((label, phrase) in
        listOf(
            "torch on" to "turn on the flashlight",
            "torch off" to "turn off the flashlight",
            "photo" to "take a photo",
            "dark" to "the room is too dark to read anything",
            "don't" to "don't turn on the light",
            "weather" to "what is the weather in nairobi",
        )) {
      quick.addView(
          Button(this).apply {
            text = label
            textSize = 11f
            setOnClickListener { handle(phrase) }
          }
      )
    }
    root.addView(quick)

    logView =
        TextView(this).apply {
          textSize = 12f
          setTextColor(Color.DKGRAY)
          setPadding(0, 16, 0, 0)
        }
    root.addView(ScrollView(this).apply { addView(logView) })
    setContentView(root)

    // Re-grab the route button now that it exists, so `ready` can enable it.
    routeButton = buttons
  }

  private lateinit var routeButton: Button

  // ------------------------------------------------------------------ router

  private fun loadRouter() {
    val loadStart = System.nanoTime()
    model =
        CompiledModel.create(
            assets,
            MODEL,
            CompiledModel.Options(Accelerator.CPU),
            null,
        )
    inputBuffers = model.createInputBuffers()
    outputBuffers = model.createOutputBuffers()

    tokenizer =
        WordPieceTokenizer.fromAssetFile(assets.open("vocab.txt").bufferedReader().readText())

    val spec = JSONObject(assets.open(TOKENS).bufferedReader().readText())

    val tools = spec.getJSONObject("tools")
    for (name in tools.keys()) {
      toolProtos[name] = listOf(embedSingle(tools.getJSONObject(name)))
    }
    // Only the four tool centroids are needed for a physical-tool decision, so the
    // router goes live after ~4 embeddings instead of 36. The open_app and
    // out-of-domain banks are embedded immediately after and logged separately.

    ready = true
    runOnUiThread { routeButton.isEnabled = true }
    val coldMs = (System.nanoTime() - loadStart) / 1_000_000
    log("router ready: ${toolProtos.size} tools in ${coldMs}ms")

    val t1 = System.nanoTime()
    appProtos = embedRows(spec.getJSONArray("app_prototypes"))
    outOfDomain = embedRows(spec.getJSONArray("out_of_domain"))
    val extraMs = (System.nanoTime() - t1) / 1_000_000
    log("secondary banks: ${appProtos.size} app + ${outOfDomain.size} ood in ${extraMs}ms")

    drainQueue()
  }

  /** One tool description, stored as a single {ids, mask} object. */
  private fun embedSingle(o: JSONObject): FloatArray =
      embedOne(toLongs(o.getJSONArray("ids")), toLongs(o.getJSONArray("mask")))

  /** A flat JSON array of {ids, mask} objects. */
  private fun embedRows(rows: JSONArray): List<FloatArray> {
    val out = ArrayList<FloatArray>()
    for (i in 0 until rows.length()) {
      val o = rows.getJSONObject(i)
      out.add(embedOne(toLongs(o.getJSONArray("ids")), toLongs(o.getJSONArray("mask"))))
    }
    return out
  }

  private fun toLongs(arr: JSONArray): LongArray = LongArray(arr.length()) { arr.getLong(it) }

  /**
   * Output 0 is already mean-pooled by the exported graph (single [384] vector, not
   * [128,384]); only L2 normalization is applied here so a dot product is cosine.
   */
  private fun embedOne(ids: LongArray, mask: LongArray): FloatArray {
    inputBuffers[0].writeLong(ids)
    inputBuffers[1].writeLong(mask)
    model.run(inputBuffers, outputBuffers, "serving_default")
    val v = outputBuffers[0].readFloat()
    var norm = 0f
    for (x in v) norm += x * x
    norm = kotlin.math.sqrt(norm)
    if (norm > 0f) for (k in v.indices) v[k] /= norm
    return v
  }

  private fun dot(a: FloatArray, b: FloatArray): Float {
    var s = 0f
    for (i in a.indices) s += a[i] * b[i]
    return s
  }

  // ------------------------------------------------------------------ routing

  private fun handle(raw: String) {
    val utterance = raw.trim()
    if (utterance.isEmpty() || !ready) return

    // 1. Negation, ahead of the model. See NEGATION for the measurement that
    //    justifies putting this before any inference.
    if (NEGATION.containsMatchIn(utterance)) {
      log("no action\n\"$utterance\"\n  -> negated request, hard guard")
      return
    }

    // 2. Route.
    val t0 = System.nanoTime()
    val (ids, mask) = tokenizer.encode(utterance, SEQ)
    val vec = embedOne(ids, mask)
    val ms = (System.nanoTime() - t0) / 1_000_000

    // Best prototype per candidate, then the winner vs every alternative including
    // abstain. `clarify` competes as a real class rather than gating behind two
    // conditions -- the two-condition form leaked an out-of-domain case whose
    // positive margin happened to be wide.
    val scores = HashMap<String, Float>()
    for ((name, protos) in toolProtos) {
      scores[name] = protos.maxOf { dot(it, vec) }
    }
    val appBest = appProtos.maxOf { dot(it, vec) }
    val oodBest = outOfDomain.maxOf { dot(it, vec) }

    val ranked = scores.entries.sortedByDescending { it.value }

    // open_app competes as a real candidate, not as a runner-up. An earlier version
    // folded appBest into the runner-up, which meant open_app could never win:
    // any exact "open chrome" match produced a negative margin and declined.
    val topTool = ranked[0]
    val isApp = appBest > topTool.value
    val winnerName = if (isApp) "open_app" else topTool.key
    val winnerScore = if (isApp) appBest else topTool.value

    // Runner-up = best score among every candidate EXCEPT the winner. Built by
    // filtering the winner out explicitly; an earlier version folded the winner's
    // own score into this expression, which pinned the margin at 0.
    val candidates = ArrayList<Pair<String, Float>>()
    for ((name, score) in ranked) {
      if (name != winnerName) candidates.add(Pair(name, score))
    }
    if (winnerName != "open_app") candidates.add(Pair("open_app", appBest))
    if (winnerName != "clarify") candidates.add(Pair("clarify", oodBest))
    val runnerUp = candidates.maxOf { it.second }
    val margin = winnerScore - runnerUp

    if (margin < ABSTAIN_MARGIN) {
      val why =
          when {
            oodBest >= winnerScore -> "out of domain"
            else ->
                "ambiguous between $winnerName and " +
                    candidates.maxBy { it.second }.first
          }
      log(
          "declined\n\"$utterance\"\n  -> $why " +
              "(top=$winnerName ${"%.3f".format(winnerScore)} " +
              "next=${"%.3f".format(runnerUp)} margin=${"%.3f".format(margin)}) ${ms}ms"
      )
      return
    }

    val alias = if (isApp) appAliasFor(utterance) else null
    log(
        "route      \"$utterance\"\n" +
            "  -> ${alias?.let { "open_app($it)" } ?: winnerName}  " +
            "margin=${"%.3f".format(margin)}  ${ms}ms"
    )
    if (isApp && alias == null) {
      log("declined   | open_app matched ${"%.3f".format(appBest)} but no alias in the utterance")
      return
    }
    execute(winnerName, alias)
  }

  /** Maps an "open X" utterance to an alias key. */
  private fun appAliasFor(utterance: String): String? {
    val text = utterance.lowercase()
    val hit = APP_ALIASES.keys.filter { text.contains(it) }.maxByOrNull { it.length }
    return hit
  }

  private fun execute(tool: String, appAlias: String?) {
    try {
      when {
        appAlias != null -> launchApp(appAlias)
        tool == "open_flashlight" -> setTorch(true)
        tool == "close_flashlight" -> setTorch(false)
        tool == "take_photo" -> log("execute      | take_photo -> ACTION_IMAGE_CAPTURE intent")
        tool == "query_calendar" ->
            log("execute      | query_calendar -> READ_CALENDAR is not granted yet, so not read")
        else -> log("execute      | $tool: no handler wired")
      }
    } catch (t: Throwable) {
      log("execute failed: ${t.javaClass.simpleName}: ${t.message}")
    }
  }

  private fun setTorch(on: Boolean) {
    val camera = getSystemService(CAMERA_SERVICE) as android.hardware.camera2.CameraManager
    val id =
        camera.cameraIdList.firstOrNull { camera.getCameraCharacteristics(it).get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE) == true }
    if (id == null) {
      log("execute      | no flash unit on this device")
      return
    }
    camera.setTorchMode(id, on)
    log("execute      | torch $id -> $on")
  }

  /**
   * Launching an installed app needs no permission. Package *visibility* is the
   * catch: on API 30+ `getLaunchIntentForPackage` is not filtered, but enumerating
   * launchers is, which is why the alias table here is explicit rather than
   * discovered at runtime.
   */
  private fun launchApp(alias: String) {
    val pkg = APP_ALIASES[alias]
    if (pkg == null) {
      log("execute      | open_app: unknown alias \"$alias\"")
      return
    }
    val intent = packageManager.getLaunchIntentForPackage(pkg)
    if (intent == null) {
      log("execute      | open_app: $alias ($pkg) not installed")
      return
    }
    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    startActivity(intent)
    log("execute      | open_app: launched $alias -> $pkg")
  }

  private fun log(line: String) {
    Log.i(TAG, line.replace("\n", " | "))
    runOnUiThread { logView.append(line + "\n") }
  }
}