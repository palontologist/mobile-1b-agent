package com.ai.edge.agent

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale
import kotlin.math.sqrt

/**
 * Phone Assistant — press-to-talk voice loop, on-device.
 *
 * Built on what was actually measured on the SM-A145F, not assumed:
 *
 *  - `MediaRecorder.AudioSource.MIC` is the only source that reliably carries audio
 *    here. Measured: MIC 71-148 RMS, VOICE_RECOGNITION 12, VOICE_COMMUNICATION 1.4 —
 *    the latter two report `recordingState == RECORDING` while yielding almost
 *    nothing. The self-test in CallAudioProbeActivity proved a 25x separation
 *    (silence 74 vs 440Hz tone 1872), so the metrics are trustworthy.
 *  - The microphone is HARD MUTED during a carrier call: measured `rms=0.0` on all
 *    three sources with `AudioManager.getMode() == MODE_IN_CALL`. This app therefore
 *    does NOT try to work during phone calls, and says so rather than pretending.
 *    See measurements/2026-10-01-callprobe/.
 *  - Half-duplex is deliberate. FunctionGemma 270M measures 21.4 tok/s with 428ms
 *    TTFT on this device — fine for a turn, useless for barge-in. Holding a button
 *    makes turn ownership a UI event instead of an inference problem.
 *
 * The loop is record -> playback -> reply. Speech-to-text is NOT wired: LiteRT-LM is
 * a text LLM and no on-device STT model has been verified on this hardware. Rather
 * than fake it, this app plays back what it captured and speaks a canned line, which
 * proves the audio path in both directions before a model is involved.
 */
class PhoneAssistantActivity : Activity() {

    companion object {
        private const val TAG = "PhoneAssistant"
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        // Measured, then corrected. Silence on this device reads 34-74 RMS; a first
        // real utterance read 229. An initial floor of 400 was WRONG: it classified
        // normal speech as silence and silently skipped STT entirely (stt_ms=0). 120
        // sits above the observed noise floor and well below observed speech, and
        // the threshold is logged with every turn so this is visible rather than
        // inferred.
        private const val RMS_FLOOR = 120.0

        private const val TTS_IDLE = android.speech.tts.TextToSpeech.QUEUE_FLUSH
    }

    private lateinit var statusText: TextView
    private lateinit var transcriptText: TextView
    private var tts: TextToSpeech? = null
    private var stt: SherpaStt? = null
    private var ttsReady = false
    private var recording = false
    private var recordingThread: Thread? = null
    @Volatile private var stopRequested = false
    private var turnSeq = 0

    /** One measured turn, kept so latency can be compared across runs. */
    private data class Turn(
        val id: Int,
        val recordMs: Long,
        val recordRms: Double,
        val playbackMs: Long,
        val ttsStartMs: Long,
        val speakMs: Long
    )

    private val turns = mutableListOf<Turn>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestPermissions()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }

        root.addView(TextView(this).apply {
            textSize = 12f
            text = buildString {
                append("Phone Assistant — press and hold to talk\n")
                append("half-duplex: you have the floor while held\n\n")
                append("Mic source: MIC (measured 71-148 RMS on this device;\n")
                append("VOICE_RECOGNITION and VOICE_COMMUNICATION measured\n")
                append("near-silent and are deliberately not used).\n\n")
                append("Not usable during a carrier phone call — the mic is\n")
                append("hard-muted then (measured rms=0.0, MODE_IN_CALL).")
            }
        })

        statusText = TextView(this).apply {
            textSize = 16f
            setPadding(0, 24, 0, 12)
            text = getString(R.string.assistant_ready)
        }
        root.addView(statusText)

        transcriptText = TextView(this).apply {
            textSize = 13f
            setPadding(0, 0, 0, 12)
            text = ""
        }
        root.addView(transcriptText)

        val talk = Button(this).apply {
            text = getString(R.string.assistant_hold_to_talk)
            // Press-and-hold rather than toggle: the floor is explicit, and a
            // half-duplex UI that can get stuck talking is worse than no UI.
            setOnTouchListener { v, event ->
                when (event.actionMasked) {
                    android.view.MotionEvent.ACTION_DOWN -> { v.isPressed = true; startCapture(); true }
                    android.view.MotionEvent.ACTION_UP, android.view.MotionEvent.ACTION_CANCEL -> {
                        v.isPressed = false
                        stopCapture()
                        true
                    }
                    else -> false
                }
            }
        }
        root.addView(talk)

        root.addView(Button(this).apply {
            text = getString(R.string.assistant_report)
            setOnClickListener { showReport() }
        })

        // --- skills panel (mirrors phone-harness helpers) ---
        val skillsHeader = TextView(this).apply {
            textSize = 12f
            setPadding(0, 28, 0, 8)
            text = if (PhoneSkillsService.isReady) {
                "skills: LIVE"
            } else {
                "skills: OFF — enable Phone Assistant skills in Settings > Accessibility"
            }
        }
        root.addView(skillsHeader)

        root.addView(Button(this).apply {
            text = getString(R.string.skill_list_apps)
            setOnClickListener { skillListApps() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.skill_read_screen)
            setOnClickListener { skillReadScreen() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.skill_open_camera)
            setOnClickListener { skillOpenApp("Camera") }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.skill_open_settings)
            setOnClickListener { skillOpenApp("Settings") }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.skill_find_tap)
            setOnClickListener { skillFindAndTap() }
        })

        setContentView(ScrollView(this).apply { addView(root) })

        Thread {
            val engine = SherpaStt(this)
            if (engine.load()) {
                stt = engine
                statusText.post { statusText.text = "ready — STT loaded (${engine.loadMs()}ms)" }
            } else {
                statusText.post { statusText.text = "ready — STT unavailable, replies stay canned" }
            }
        }.also { it.isDaemon = true; it.start() }

        tts = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                tts?.language = Locale.US
                Log.i(TAG, "tts_ready engine=${tts?.defaultEngine}")
            } else {
                Log.w(TAG, "tts_unavailable status=$status")
            }
        }

        Log.i(TAG, "activity_created sdk=${android.os.Build.VERSION.SDK_INT} perms=${permSummary()}")
    }

    private fun permSummary(): String = listOf(
        Manifest.permission.RECORD_AUDIO to "RECORD_AUDIO"
    ).joinToString(",") { (p, n) -> if (checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED) n else "-$n" }

    private fun requestPermissions() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }
    }

    /**
     * Record until [stopCapture], then run the whole turn.
     *
     * Runs off the main thread: AudioRecord.read() blocks, and doing that on the UI
     * thread is how probe apps end up dropping frames or ANRing.
     */
    private fun startCapture() {
        if (recording) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            statusText.text = "needs RECORD_AUDIO"
            return
        }
        recording = true
        stopRequested = false
        val t0 = SystemClock.elapsedRealtime()

        recordingThread = Thread {
            val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
            if (minBuf <= 0) {
                Log.w(TAG, "capture_unsupported minBuf=$minBuf")
                statusText.post { statusText.text = "mic unsupported" }
                recording = false
                return@Thread
            }
            var rec: AudioRecord? = null
            try {
                rec = AudioRecord(
                    MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL, ENCODING,
                    maxOf(minBuf * 4, SAMPLE_RATE / 2)
                )
                if (rec.state != AudioRecord.STATE_INITIALIZED) {
                    Log.w(TAG, "capture_not_initialized state=${rec.state}")
                    statusText.post { statusText.text = "mic not initialised" }
                    recording = false
                    return@Thread
                }
                rec.startRecording()
                val state = rec.recordingState
                val out = File(cacheDir, "assistant-turn-${turnSeq + 1}.pcm")
                val buf = ShortArray(maxOf(minBuf * 4, SAMPLE_RATE / 2) / 2)
                out.outputStream().use { sink ->
                    val bytes = ByteArray(buf.size * 2)
                    var sumSq = 0.0
                    var count = 0L
                    while (!stopRequested) {
                        val n = rec.read(buf, 0, buf.size)
                        if (n <= 0) continue
                        for (i in 0 until n) {
                            val v = buf[i].toInt()
                            sink.write(bytes, i * 2, 2)
                            val s = v.toShort().toDouble()
                            sumSq += s * s
                            count++
                        }
                    }
                    val rms = if (count == 0L) 0.0 else sqrt(sumSq / count)
                    Log.i(
                        TAG,
                        "capture_done recState=$state samples=$count rms=${"%.1f".format(rms)} " +
                            "record_ms=${SystemClock.elapsedRealtime() - t0} file=${out.absolutePath}"
                    )
                    finishTurn(t0, rms, out)
                }
            } catch (e: Exception) {
                Log.e(TAG, "capture_failed ${e.message}", e)
                statusText.post { statusText.text = "capture failed: ${e.message}" }
            } finally {
                try { rec?.stop() } catch (_: Exception) {}
                rec?.release()
                recording = false
                statusText.post { if (statusText.text == getString(R.string.assistant_listening)) statusText.text = getString(R.string.assistant_ready) }
            }
        }.also { it.isDaemon = true; it.start() }

        statusText.text = getString(R.string.assistant_listening)
    }

    private fun stopCapture() {
        if (!recording) return
        stopRequested = true
    }

    /**
     * Play back what was captured, then speak a canned reply.
     *
     * Playback first is deliberate: it proves the mic captured the user's actual
     * voice before anything downstream can be blamed for a bad transcript. And it is
     * the honest placeholder for STT until an on-device speech model is verified.
     */
    private fun finishTurn(t0: Long, rms: Double, pcm: File) {
        val id = ++turnSeq
        val recordMs = SystemClock.elapsedRealtime() - t0

        // Play the raw capture back. 16k mono PCM16 needs a WAV header to be
        // playable, so wrap it rather than asking MediaPlayer to guess.
        val wav = File(cacheDir, "assistant-turn-$id.wav")
        writeWav(pcm, wav)
        val p0 = SystemClock.elapsedRealtime()
        playBlocking(wav)
        val playbackMs = SystemClock.elapsedRealtime() - p0

        val heardSomething = rms >= RMS_FLOOR
        statusText.post {
            statusText.text = if (heardSomething) {
                "turn $id · heard you (rms ${"%.0f".format(rms)}) · ${recordMs}ms capture + ${playbackMs}ms playback"
            } else {
                "turn $id · silence (rms ${"%.0f".format(rms)}) · ${recordMs}ms"
            }
        }

        // Decode what was actually said. This is the piece that turns a canned
        // reply into a real one; empty result is common on an int8 transducer and is
        // NOT an error, so it gets its own wording rather than the error branch.
        val decodeStart = SystemClock.elapsedRealtime()
        val transcript = if (heardSomething) {
            val raw = readAllBytes(pcm)
            stt?.transcribe(raw).orEmpty()
        } else {
            ""
        }
        val decodeMs = SystemClock.elapsedRealtime() - decodeStart

        statusText.post {
            statusText.text = if (heardSomething) {
                "turn $id · heard (rms ${"%.0f".format(rms)}) · STT ${decodeMs}ms · \"$transcript\""
            } else {
                "turn $id · silence (rms ${"%.0f".format(rms)}) · ${recordMs}ms"
            }
        }

        val reply = when {
            !heardSomething -> "I did not hear anything. Hold the button and speak while it is held."
            transcript.isEmpty() -> "I heard something but could not make out any words."
            else -> "You said: $transcript"
        }

        // Measured from the end of capture, so this is a real latency rather than an
        // absolute clock reading (which is what a bare elapsedRealtime() would give).
        val ttsStart = SystemClock.elapsedRealtime()
        speakBlocking(reply)
        val speakMs = SystemClock.elapsedRealtime() - ttsStart

        turns.add(Turn(id, recordMs, rms, playbackMs, ttsStart, speakMs))
        Log.i(
            TAG,
            "turn_end id=$id record_ms=$recordMs rms=${"%.1f".format(rms)} " +
                "playback_ms=$playbackMs stt_ms=$decodeMs reply_wait_ms=$speakMs " +
                "heard=$heardSomething transcript=\"${transcript.take(60)}\" " +
                "end_to_end_ms=${SystemClock.elapsedRealtime() - t0}"
        )
        // Cache files accumulate one pair per turn; this is a probe, not a feature,
        // but an unbounded cache dir is still a bug on a 3.5GB device.
        pcm.delete()
        wav.delete()
    }

    private fun readAllBytes(f: File): ByteArray = if (f.exists()) f.readBytes() else ByteArray(0)

    private fun writeWav(pcmFile: File, wavFile: File) {
        val dataSize = pcmFile.length().toInt()
        val byteRate = SAMPLE_RATE * 2
        val header = java.io.ByteArrayOutputStream()
        fun ascii(s: String) = header.write(s.toByteArray(Charsets.US_ASCII))
        fun le32(v: Int) {
            header.write(v and 0xFF); header.write((v shr 8) and 0xFF)
            header.write((v shr 16) and 0xFF); header.write((v shr 24) and 0xFF)
        }
        fun le16(v: Int) { header.write(v and 0xFF); header.write((v shr 8) and 0xFF) }
        ascii("RIFF"); le32(36 + dataSize); ascii("WAVE")
        ascii("fmt "); le32(16); le16(1); le16(1)
        le32(SAMPLE_RATE); le32(byteRate); le16(2); le16(16)
        ascii("data"); le32(dataSize)
        wavFile.outputStream().use { it.write(header.toByteArray()); it.write(pcmFile.readBytes()) }
    }

    private fun playBlocking(wav: File) {
        try {
            val mp = MediaPlayer()
            mp.setDataSource(wav.absolutePath)
            mp.prepare()
            mp.start()
            while (mp.isPlaying) { Thread.sleep(40) }
            mp.release()
        } catch (e: Exception) {
            Log.w(TAG, "playback_failed ${e.message}")
        }
    }

    private fun speakBlocking(text: String) {
        val engine = tts ?: return
        if (!ttsReady) { Log.w(TAG, "speak_skipped tts_not_ready"); return }
        engine.speak(text, TTS_IDLE, null, "turn-${SystemClock.elapsedRealtime()}")
        val t0 = SystemClock.elapsedRealtime()
        while (engine.isSpeaking && SystemClock.elapsedRealtime() - t0 < 20_000) {
            Thread.sleep(50)
        }
    }

    // --- skill dispatch -------------------------------------------------------
    // Each wrapper exists so the UI never touches the service directly: an absent or
    // disabled accessibility service must degrade to a message, never a crash.

    private fun skills(): PhoneSkillsService? =
        PhoneSkillsService.instance.takeIf { PhoneSkillsService.isReady }

    private fun skillListApps() {
        val svc = skills() ?: run { statusText.text = "skills not enabled"; return }
        val apps = svc.listApps()
        val preview = apps.take(12).joinToString(", ") { it.first }
        statusText.text = "${apps.size} apps: $preview${if (apps.size > 12) " …" else ""}"
    }

    private fun skillReadScreen() {
        val svc = skills() ?: run { statusText.text = "skills not enabled"; return }
        val nodes = svc.readScreen()
        val preview = nodes.take(8).joinToString(" | ") { it.text.take(24) }
        statusText.text = if (nodes.isEmpty()) {
            "screen: no readable nodes"
        } else {
            "${nodes.size} nodes: $preview"
        }
    }

    private fun skillOpenApp(label: String) {
        val svc = skills() ?: run { statusText.text = "skills not enabled"; return }
        statusText.text = if (svc.openApp(label)) "opened $label" else "no app matched $label"
    }

    /** The phone-harness shape: find a label, then tap the point it resolved to. */
    private fun skillFindAndTap() {
        val svc = skills() ?: run { statusText.text = "skills not enabled"; return }
        val target = statusText.text?.takeIf { it.isNotBlank() }?.split("  ")?.firstOrNull() ?: "Settings"
        val point = svc.findText(target)
        if (point == null) {
            statusText.text = "find_text('$target') not found"
        } else {
            statusText.text = if (svc.tap(point.x, point.y)) "tapped $target at (${point.x},${point.y})" else "tap rejected"
        }
    }

    private fun showReport() {
        if (turns.isEmpty()) {
            statusText.text = "no turns yet"
            return
        }
        val text = buildString {
            append("turns: ${turns.size}\n")
            append("median record ms : ${turns.map { it.recordMs }.sorted()[turns.size / 2]}\n")
            append("median playback ms: ${turns.map { it.playbackMs }.sorted()[turns.size / 2]}\n")
            append("median speak ms  : ${turns.map { it.speakMs }.sorted()[turns.size / 2]}\n")
            append("peak rms         : ${"%.0f".format(turns.maxOf { it.recordRms })}\n")
        }
        statusText.text = text
        Log.i(TAG, "report $text")
    }
}