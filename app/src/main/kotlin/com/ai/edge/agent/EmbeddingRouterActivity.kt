package com.ai.edge.agent

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * On-device test of the embedding router that replaces FunctionGemma's tool routing.
 *
 * The desktop measurement (`scripts/measure_embedding_router.py`) gave 18/18 at
 * 32 ms median on x86-64. This runs the same four tool descriptions and the same
 * 18 prompts on the phone through the LiteRT CompiledModel API, and records the two
 * numbers a desktop run cannot give: device latency and peak RSS.
 *
 * Tokenization is pre-computed on the host (`minilm_tokens.json` in assets) because
 * the Python tokenizer does not run here. The ids are byte-identical to what the
 * desktop harness produced, so both runs see the same input.
 *
 * Launch: `adb shell am start -n com.ai.edge.agent/.EmbeddingRouterActivity`
 */
class EmbeddingRouterActivity : Activity() {

  companion object {
    private const val TAG = "EMR"
    private const val MODEL = "minilm.tflite"
    private const val TOKENS = "minilm_tokens.json"
    private const val SEQ = 128
    private const val DIM = 384
    private const val ROUNDS = 3
  }

  private lateinit var model: CompiledModel
  private lateinit var inputBuffers: List<TensorBuffer>
  private lateinit var outputBuffers: List<TensorBuffer>

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    Thread {
      try {
        run()
      } catch (t: Throwable) {
        Log.e(TAG, "PROBE_FAILED ${t.javaClass.name}: ${t.message}", t)
      }
    }.start()
  }

  /** VmHWM is the kernel's peak-RSS watermark, so it survives a GC mid-run. */
  private fun peakRssMb(): Long =
      File("/proc/self/status")
          .readLines()
          .first { it.startsWith("VmHWM:") }
          .split(Regex("\\s+"))[1]
          .toLong() / 1024

  private fun dot(a: FloatArray, b: FloatArray): Float {
    var sum = 0f
    for (i in a.indices) sum += a[i] * b[i]
    return sum
  }

  /**
   * Mean-pools over real tokens then L2-normalizes, matching the desktop harness.
   *
   * Inputs 0 and 1 are `input_ids` and `attention_mask`, both int64 [1,128]; output 0
   * is an already-pooled [1,384] float32 vector. Buffers are reused across calls, so
   * the per-prompt cost is inference only.
   */
  private fun embed(ids: LongArray, mask: LongArray): FloatArray {
    inputBuffers[0].writeLong(ids)
    inputBuffers[1].writeLong(mask)
    // By signature NAME, not index. The Python binding treats 1 as
    // "serving_default"; this graph has exactly one signature (index 0), so the
    // numeric overload fails with "Signature index is out of range of signature
    // keys". Named dispatch is version-proof against that difference.
    model.run(inputBuffers, outputBuffers, "serving_default")

    // Output 0 is the pooled [384] sentence vector -- the sentence-transformers
    // export bakes mean pooling into the graph, so it is NOT [128, 384] and must
    // not be pooled again on this side. L2 normalize so a later dot product is a
    // cosine similarity.
    val vec = outputBuffers[0].readFloat()
    var norm = 0f
    for (v in vec) norm += v * v
    norm = kotlin.math.sqrt(norm)
    if (norm > 0f) for (i in vec.indices) vec[i] /= norm
    return vec
  }

  private fun toLongs(arr: JSONArray): LongArray = LongArray(arr.length()) { arr.getLong(it) }

  private fun encoded(o: JSONObject): Pair<LongArray, LongArray> =
      toLongs(o.getJSONArray("ids")) to toLongs(o.getJSONArray("mask"))

  private fun run() {
    Log.i(
        TAG,
        "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT} " +
            "rss_before_mb=${peakRssMb()}"
    )

    model =
        CompiledModel.create(
            assets,
            MODEL,
            CompiledModel.Options(Accelerator.CPU),
            null,
        )
    inputBuffers = model.createInputBuffers()
    outputBuffers = model.createOutputBuffers()
    Log.i(
        TAG,
        "MODEL_READY inputs=${inputBuffers.size} outputs=${outputBuffers.size} " +
            "rss_mb=${peakRssMb()}"
    )

    val spec = JSONObject(assets.open(TOKENS).bufferedReader().readText())

    val toolVecs = LinkedHashMap<String, FloatArray>()
    val tools = spec.getJSONObject("tools")
    for (name in tools.keys()) {
      val (ids, mask) = encoded(tools.getJSONObject(name))
      toolVecs[name] = embed(ids, mask)
    }
    Log.i(TAG, "TOOLS_READY n=${toolVecs.size} rss_mb=${peakRssMb()}")

    val probes = spec.getJSONArray("probes")
    for (round in 0 until ROUNDS) {
      var hits = 0
      val hitsByKind = HashMap<String, Int>()
      val nByKind = HashMap<String, Int>()
      val times = ArrayList<Long>()

      for (i in 0 until probes.length()) {
        val p = probes.getJSONObject(i)
        val prompt = p.getString("prompt")
        val expected = p.getString("expected")
        val kind = p.getString("kind")
        val (ids, mask) = encoded(p.getJSONObject("enc"))

        val t0 = System.nanoTime()
        val vec = embed(ids, mask)
        val ms = (System.nanoTime() - t0) / 1_000_000
        times.add(ms)

        val ranked = toolVecs.entries.sortedByDescending { dot(it.value, vec) }
        val got = ranked[0].key
        val margin = dot(ranked[0].value, vec) - dot(ranked[1].value, vec)
        val ok = got == expected
        if (ok) hits++
        nByKind[kind] = (nByKind[kind] ?: 0) + 1
        if (ok) hitsByKind[kind] = (hitsByKind[kind] ?: 0) + 1

        Log.i(
            TAG,
            "ROW $round|$kind|$expected|$got|$ok|$ms|${"%.4f".format(margin)}|$prompt"
        )
      }

      val sorted = times.sorted()
      val median = sorted[sorted.size / 2]
      val kinds =
          nByKind.keys.sorted().joinToString(" ") {
              "$it=${hitsByKind[it] ?: 0}/${nByKind[it]}"
          }
      Log.i(
          TAG,
          "ROUND_SUMMARY round=$round scored=${probes.length()} hits=$hits " +
              "acc=$hits/${probes.length()} median_ms=$median min_ms=${sorted.first()} " +
              "max_ms=${sorted.last()} peak_rss_mb=${peakRssMb()} kinds[$kinds]"
      )
    }

    model.close()
    Log.i(TAG, "DONE")
  }
}