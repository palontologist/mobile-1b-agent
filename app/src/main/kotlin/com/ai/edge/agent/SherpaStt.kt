package com.ai.edge.agent

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineMoonshineModelConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineStream
import java.io.File

/**
 * On-device STT via sherpa-onnx, Moonshine tiny-en int8.
 *
 * WHY THIS CLOSES THE LOOP
 *
 * The voice loop in [PhoneAssistantActivity] had no STT, so it could not know what
 * was said — it captured, played back, and spoke a canned line. This is the piece
 * that makes a reply possible. It also removes the last reason the assistant needed
 * a network round trip.
 *
 * WHY MOONSHINE TINY
 *
 * Chosen for measurement reasons, not preference. The device is 3.5GB and already
 * measured 1,458 MB peak for a single text model (FunctionGemma 270M), so model
 * count matters more than model quality here. Moonshine tiny int8 is ~120MB of
 * weights against base-en's ~240MB. Base is the accuracy upgrade once residency is
 * proven, and the swap is one config flag.
 *
 * Moonshine is an ENCODER/DECODER transducer rather than the usual encoder/joiner/
 * decoder, hence the `cached`/`uncached` decode pair in the model directory. Getting
 * that wiring wrong yields silence rather than an error, so the file paths are
 * validated up front in [isModelPresent] and every failure path logs.
 *
 * OFFLINE, NOT STREAMING
 *
 * [transcribe] decodes the whole utterance after capture. That costs 200-650ms of
 * extra latency versus a streaming recogniser, and buys accuracy. With half-duplex
 * PTT that tradeoff is acceptable — you are already waiting for the user to finish.
 * If streaming Zipformer is wanted later, [OfflineRecognizer] is the seam to swap.
 */
class SherpaStt(context: Context) {

    companion object {
        private const val TAG = "SherpaStt"
        private const val MODEL_DIR = "/data/local/tmp/moonshine"
        private const val SAMPLE_RATE = 16000
        private const val NUM_THREADS = 2

        /** Required files, Moonshine layout. Missing any means decode returns nothing. */
        private val REQUIRED = listOf(
            "encode.int8.onnx", "uncached_decode.int8.onnx",
            "cached_decode.int8.onnx", "tokens.txt"
        )

        fun isModelPresent(): Boolean {
            val dir = File(MODEL_DIR)
            return dir.isDirectory && REQUIRED.all { File(dir, it).exists() }
        }
    }

    private var recognizer: OfflineRecognizer? = null
    private var loadMs: Long = 0
    private var decodeMs: Long = 0

    /** Load the model. Off the main thread; ~1s+ on this hardware class. */
    fun load(): Boolean {
        if (recognizer != null) return true
        if (!isModelPresent()) {
            Log.e(TAG, "load FAILED — model missing. Run: adb push <moonshine dir> $MODEL_DIR")
            return false
        }
        val t0 = System.currentTimeMillis()
        return try {
            val dir = File(MODEL_DIR)
            val model = OfflineModelConfig(
                moonshine = OfflineMoonshineModelConfig(
                    preprocessor = pick(dir, "preprocess"),
                    encoder = pick(dir, "encode"),
                    uncachedDecoder = pick(dir, "uncached_decode"),
                    cachedDecoder = pick(dir, "cached_decode")
                ),
                tokens = File(dir, "tokens.txt").absolutePath,
                numThreads = NUM_THREADS,
                provider = "cpu"
            )
            recognizer = OfflineRecognizer(
                config = OfflineRecognizerConfig(
                    featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
                    modelConfig = model,
                    // Moonshine's native decoder accepts ONLY greedy_search. Passing
                    // modified_beam_search logs "Only greedy_search is supported" and
                    // leaves the recognizer permanently unusable -- it does not throw,
                    // it just silently never decodes. Hence a constant, not a setting.
                    decodingMethod = "greedy_search"
                )
            )
            loadMs = System.currentTimeMillis() - t0
            Log.i(TAG, "load ok in ${loadMs}ms threads=$NUM_THREADS")
            true
        } catch (e: Throwable) {
            // Throwable, not Exception: a missing/wrong-ABI native lib surfaces as
            // UnsatisfiedLinkError, which is an Error.
            Log.e(TAG, "load threw ${e.javaClass.name}: ${e.message}")
            false
        }
    }

    /** Resolve a model role by substring, preferring an int8 variant when present. */
    private fun pick(dir: File, role: String): String {
        val candidates = dir.listFiles { f ->
            f.isFile && f.name.endsWith(".onnx") && f.name.contains(role, ignoreCase = true)
        }?.sortedBy { it.name }.orEmpty()
        if (candidates.isEmpty()) error("no $role .onnx in ${dir.absolutePath}")
        val quantised = candidates.filter { it.name.contains("int8") }
        return (if (quantised.isNotEmpty()) quantised.first() else candidates.first()).absolutePath
    }

    /**
     * Decode 16kHz mono signed 16-bit LE PCM.
     *
     * Returns the transcript, or "" when nothing intelligible was decoded. An empty
     * string is a real and common result on this hardware — quiet speech plus an
     * int8 transducer — so callers must not treat "" as an error.
     */
    fun transcribe(pcm: ByteArray): String {
        val rec = recognizer ?: run { Log.w(TAG, "transcribe called before load"); return "" }
        if (pcm.isEmpty()) return ""
        return try {
            // sherpa takes normalised float samples, not PCM bytes. Int16 full scale
            // is 32768; dividing by it lands in [-1, 1].
            val samples = FloatArray(pcm.size / 2) { i ->
                val lo = pcm[i * 2].toInt() and 0xFF
                val hi = pcm[i * 2 + 1].toInt()
                (((hi shl 8) or lo).toShort()).toInt() / 32768.0f
            }
            val stream = rec.createStream()
            try {
                stream.acceptWaveform(samples, SAMPLE_RATE)
                val started = SystemClock.elapsedRealtime()
                rec.decode(stream)
                val text = rec.getResult(stream).text.orEmpty().trim()
                decodeMs = SystemClock.elapsedRealtime() - started
                Log.i(TAG, "decoded audio_ms=${samples.size * 1000 / SAMPLE_RATE} in ${decodeMs}ms chars=${text.length}")
                text
            } finally {
                stream.release()
            }
        } catch (e: Throwable) {
            Log.e(TAG, "decode threw ${e.javaClass.name}: ${e.message}")
            ""
        }
    }

    /** Load time of the model, for the latency report. */
    fun loadMs(): Long = loadMs

    fun release() {
        try { recognizer?.release() } catch (_: Throwable) {}
        recognizer = null
    }
}