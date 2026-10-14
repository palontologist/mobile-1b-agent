/**
 * CallAudioProbeActivity — does a third-party app actually hear the caller?
 *
 * The question this answers is binary and decides whether an Omi agent can ever
 * talk on a carrier phone call: the app can *answer* the call (TelecomManager,
 * ANSWER_PHONE_CALLS), and it can *speak* through the speaker, but can it *capture*
 * the caller's voice?
 *
 * Two Android facts make this uncertain, not obviously no:
 *
 *  - `MediaRecorder.AudioSource.VOICE_CALL` and `VOICE_DOWNLINK` are the sources that
 *    would carry the caller, and both require `CAPTURE_AUDIO_OUTPUT` — a
 *    signature|privileged permission a normal app cannot hold.
 *  - `MIC` / `VOICE_RECOGNITION` / `VOICE_COMMUNICATION` need no special permission,
 *    but telephony holds `MODE_IN_CALL` during an active call and may starve them.
 *
 * Neither has been measured here, so this probe measures it. Consistent with the
 * torch probe's rule — "the single-press happy path proves nothing" — success is
 * judged by **measured RMS amplitude**, never by `AudioRecord` merely opening.
 * `recordingState == RECORDSTATE_RECORDING` is reported alongside precisely so a
 * silent-but-"recording" stream is visible as such.
 *
 * Usage (device must be the SM-A145F on a real carrier call, or Wi-Fi calling):
 *
 *   adb logcat -v epoch CallProbe:V '*:S'
 *   adb shell am start -n com.ai.edge.agent/.CallAudioProbeActivity
 *
 * Call from a second phone on speaker, press ANSWER, then RUN. Read the RMS.
 */
package com.ai.edge.agent

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.ToneGenerator
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.telecom.TelecomManager
import android.telephony.TelephonyCallback
import android.telephony.TelephonyManager
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

class CallAudioProbeActivity : Activity() {

    companion object {
        private const val TAG = "CallProbe"
        private const val SAMPLE_RATE = 16000
        private const val MEASURE_MS = 4000
        private const val HOLD_MS = 45_000
        private const val MONITOR_MS = 90_000

        /**
         * Audio sources under test, best candidate first.
         *
         * `CAPTURE_AUDIO_OUTPUT` is signature|privileged, so VOICE_CALL and
         * VOICE_DOWNLINK are expected to fail here. They are included anyway: a
         * device or OEM that permits them changes the answer, and that is worth
         * discovering rather than assuming.
         */
        private val SOURCES = listOf(
            "VOICE_COMMUNICATION" to MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            "VOICE_RECOGNITION" to MediaRecorder.AudioSource.VOICE_RECOGNITION,
            "MIC" to MediaRecorder.AudioSource.MIC,
            "VOICE_CALL" to MediaRecorder.AudioSource.VOICE_CALL,
            "VOICE_DOWNLINK" to MediaRecorder.AudioSource.VOICE_DOWNLINK
        )
    }

    private lateinit var statusText: TextView
    private val callActive = AtomicBoolean(false)

    @Volatile
    private var callStateName: String = "unknown"
    private var telemetry: TelephonyManager? = null

    private val callStateCallback = object : TelephonyCallback(), TelephonyCallback.CallStateListener {
        override fun onCallStateChanged(state: Int) {
            val name = when (state) {
                TelephonyManager.CALL_STATE_IDLE -> "idle"
                TelephonyManager.CALL_STATE_RINGING -> "ringing"
                TelephonyManager.CALL_STATE_OFFHOOK -> "offhook"
                else -> "unknown($state)"
            }
            // offhook means the call is up and the audio path is owned by telephony.
            callActive.set(state == TelephonyManager.CALL_STATE_OFFHOOK)
            callStateName = name
            Log.i(TAG, "[CALLPROBE] call_state=$name")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        telemetry = getSystemService(Context.TELEPHONY_SERVICE) as? TelephonyManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && telemetry != null) {
            try {
                telemetry!!.registerTelephonyCallback(mainExecutor, callStateCallback)
            } catch (e: SecurityException) {
                Log.w(TAG, "[CALLPROBE] cannot register call-state callback: ${e.message}")
            }
        }

        requestPermissions()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }

        root.addView(TextView(this).apply {
            textSize = 12f
            text = buildString {
                append("device : ${Build.MANUFACTURER} ${Build.MODEL}\n")
                append("sdk    : ${Build.VERSION.SDK_INT}\n")
                append("audio  : ${permissionsSummary()}\n")
                append("\nCall from a second phone on SPEAKER first.\n")
                append("Then: RUN. Read RMS per source.\n")
                append("RMS ~= 0 on every source => agent cannot hear the caller.")
            }
        })

        statusText = TextView(this).apply {
            textSize = 14f
            setPadding(0, 24, 0, 24)
            text = getString(R.string.callprobe_ready)
        }
        root.addView(statusText)

        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_answer)
            setOnClickListener { answer() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_selftest)
            setOnClickListener { requestPermissions(); selfTest() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_incallmode)
            setOnClickListener { forceInCallMode() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_monitor)
            setOnClickListener { monitor() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_holdmic)
            setOnClickListener { holdMic() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_normalmode)
            setOnClickListener { restoreNormalMode() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_measure)
            setOnClickListener { measureAll() }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.callprobe_btn_run)
            setOnClickListener { runAll() }
        })

        setContentView(ScrollView(this).apply { addView(root) })
        Log.i(TAG, "[CALLPROBE] activity created sdk=${Build.VERSION.SDK_INT} perms=${permissionsSummary()}")
    }

    private fun permissionsSummary(): String = REQUIRED.joinToString(",") {
        if (checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED) {
            it.substringAfterLast('.')
        } else {
            "-${it.substringAfterLast('.')}"
        }
    }

    private val REQUIRED = arrayOf(
        Manifest.permission.ANSWER_PHONE_CALLS,
        Manifest.permission.READ_PHONE_STATE,
        Manifest.permission.RECORD_AUDIO
    )

    private fun requestPermissions() {
        val missing = REQUIRED.filter {
            checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) return
        Log.i(TAG, "[CALLPROBE] requesting ${missing.map { it.substringAfterLast('.') }}")
        requestPermissions(missing.toTypedArray(), 1)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val granted = grantResults.filter { it == PackageManager.PERMISSION_GRANTED }.size
        Log.i(
            TAG,
            "[CALLPROBE] permissions $granted/${grantResults.size} granted " +
                "summary=${permissionsSummary()}"
        )
    }

    /**
     * Answer a ringing carrier call. Needs ANSWER_PHONE_CALLS and API 26+.
     */
    private fun answer() {
        val tm = getSystemService(Context.TELECOM_SERVICE) as? TelecomManager
        if (tm == null) {
            statusText.text = "no TelecomManager"
            return
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            statusText.text = "answerRingingCall needs API 26+"
            Log.w(TAG, "[CALLPROBE] answer unsupported sdk=${Build.VERSION.SDK_INT}")
            return
        }
        val t0 = SystemClock.elapsedRealtimeNanos()
        try {
            // Deprecated in API 30, when the recommended path became posting your own
            // incoming-call notification and handling the answer yourself. That path
            // requires a ConnectionService, which in turn requires MANAGE_OWN_CALLS —
            // a permission this probe deliberately does not hold, because the question
            // here is what an ordinary app can do, not what an OMI-default app can do.
            // acceptRingingCall() still works for apps holding ANSWER_PHONE_CALLS.
            @Suppress("DEPRECATION")
            tm.acceptRingingCall()
            val us = (SystemClock.elapsedRealtimeNanos() - t0) / 1000
            Log.i(TAG, "[CALLPROBE] answer requested latency_us=$us")
            statusText.text = "answer requested"
        } catch (e: SecurityException) {
            Log.w(TAG, "[CALLPROBE] answer denied ${e.message}")
            statusText.text = "answer denied (needs ANSWER_PHONE_CALLS)"
        }
    }

    /**
     * Open [source], record for MEASURE_MS, and report peak/mean RMS.
     *
     * RMS is the only trustworthy signal. `startRecording()` succeeding and
     * `recordingState` reading RECORDING both happen on a stream that yields pure
     * silence, which is exactly the case that would otherwise be reported as a pass.
     */
    private fun measure(sourceName: String, source: Int): Double {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            Log.w(TAG, "[CALLPROBE] src=$sourceName unsupported minBuf=$minBuf")
            return -1.0
        }
        val bufBytes = maxOf(minBuf * 4, SAMPLE_RATE / 4)

        var rec: AudioRecord? = null
        return try {
            rec = AudioRecord(
                source, SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                bufBytes
            )
            if (rec.state != AudioRecord.STATE_INITIALIZED) {
                Log.w(TAG, "[CALLPROBE] src=$sourceName not_initialized state=${rec.state}")
                statusText.text = "$sourceName: not initialized"
                -1.0
            } else {
                rec.startRecording()
                val state = rec.recordingState
                val buf = ShortArray(bufBytes / 2)
                var peak = 0.0
                var sumSq = 0.0
                var count = 0L
                val deadline = SystemClock.elapsedRealtime() + MEASURE_MS
                while (SystemClock.elapsedRealtime() < deadline) {
                    val n = rec.read(buf, 0, buf.size)
                    if (n <= 0) continue
                    for (i in 0 until n) {
                        val v = buf[i].toDouble()
                        sumSq += v * v
                        count++
                        val a = abs(v)
                        if (a > peak) peak = a
                    }
                    // Per-second heartbeat so speech vs silence is visible over time.
                    Log.i(TAG, "[CALLPROBE] src=$sourceName t_ms=${MEASURE_MS - (deadline - SystemClock.elapsedRealtime())} peak=$peak")
                }
                val rms = if (count == 0L) 0.0 else sqrt(sumSq / count)
                Log.i(
                    TAG,
                    "[CALLPROBE] src=$sourceName recState=$state samples=$count " +
                        "rms=${"%.1f".format(rms)} peak=${"%.0f".format(peak)} " +
                        "VERDICT=${verdict(rms)}"
                )
                statusText.text =
                    "$sourceName  rms=${"%.0f".format(rms)}  peak=${"%.0f".format(peak)}  ${verdict(rms)}"
                rms
            }
        } catch (e: Exception) {
            // SecurityException for the privileged sources is an EXPECTED result here
            // and is itself the finding, so it is logged at info, not error.
            Log.i(TAG, "[CALLPROBE] src=$sourceName FAILED ${e.javaClass.simpleName}: ${e.message}")
            statusText.text = "$sourceName: ${e.javaClass.simpleName}"
            -1.0
        } finally {
            try {
                rec?.stop()
            } catch (_: Exception) {
            }
            rec?.release()
        }
    }

    /**
     * RMS thresholds. 32767 is full scale for PCM16; speech on a phone speaker at
     * normal distance lands well above 300, and a stream that returns pure zeros
     * sits below 1. The gap between them is enormous, so the verdict is not delicate.
     */
    private fun verdict(rms: Double): String = when {
        rms < 0 -> "UNAVAILABLE (expected for VOICE_CALL/VOICE_DOWNLINK: needs CAPTURE_AUDIO_OUTPUT)"
        rms < 1.0 -> "SILENT — stream opens but yields nothing"
        rms < 300.0 -> "WEAK — below normal speech level, do not rely on it"
        else -> "USABLE — speech-range amplitude"
    }

    private fun measureAll() {
        // Log the audio regime with every run. Without this, a reading is only
        // interpretable if you happen to remember what else was happening — and the
        // whole point is comparing MODE_NORMAL against MODE_IN_CALL.
        val am = getSystemService(Context.AUDIO_SERVICE) as? android.media.AudioManager
        Log.i(
            TAG,
            "[CALLPROBE] === MEASURE START callActive=${callActive.get()} " +
                "audioMode=${am?.mode} callState=${callStateName} " +
                "perms=${permissionsSummary()} ==="
        )
        for ((name, source) in SOURCES) {
            measure(name, source)
        }
        Log.i(
            TAG,
            "[CALLPROBE] === MEASURE END audioMode=${am?.mode} callActive=${callActive.get()} ==="
        )
    }

    /**
     * Calibration: prove the mic + RMS pipeline works at all, before blaming Android.
     *
     * Two phases. Phase A records the room in silence and must read near zero.
     * Phase B plays a loud tone through the speaker while recording, and must read
     * in speech range — that is the acoustic loop working.
     *
     * This is the control the whole probe rests on. Without it, a zero reading
     * during a call is ambiguous between "Android blocked the mic" and "this probe
     * is measuring nothing". With it, a silent phase B alongside a healthy one
     * above means the platform is the cause.
     */
    private fun selfTest() {
        Log.i(TAG, "[CALLPROBE] === SELFTEST START ===")

        Log.i(TAG, "[CALLPROBE] self_test phase=A (silence) source=MIC")
        val silent = measure("SELFTEST_A_silence", MediaRecorder.AudioSource.MIC)

        // ToneGenerator is used rather than TTS so the level is known and repeatable.
        Log.i(TAG, "[CALLPROBE] self_test phase=B (440Hz @ full volume through speaker)")
        var tone: ToneGenerator? = null
        val loud = try {
            tone = ToneGenerator(
                android.media.AudioManager.STREAM_MUSIC,
                ToneGenerator.MAX_VOLUME
            )
            tone.startTone(
                android.media.ToneGenerator.TONE_CDMA_HIGH_L, MEASURE_MS + 600
            )
            val r = measure("SELFTEST_B_tone", MediaRecorder.AudioSource.MIC)
            r
        } finally {
            try {
                tone?.stopTone()
                tone?.release()
            } catch (_: Exception) {
            }
        }

        val pipelineOk = silent >= 0 && loud > 300.0
        Log.i(
            TAG,
            "[CALLPROBE] === SELFTEST END silence=$silent tone=$loud " +
                "PIPELINE=${if (pipelineOk) "OK — mic+metrics work, a zero during a call is the platform" else "BROKEN — do not trust a call measurement until this passes"} ==="
        )
        statusText.text = if (pipelineOk) {
            "self-test OK — silence=$silent tone=$loud"
        } else {
            "self-test BROKEN — silence=$silent tone=$loud"
        }
    }

    /**
     * Try to put the audio subsystem into MODE_IN_CALL without a carrier call.
     *
     * This device has no SIM (`gsm.sim.state=ABSENT`, `IsVoiceCallAvailable=false`),
     * so a real cellular call — and therefore a genuine MODE_IN_CALL — is impossible
     * here. This is the honest approximation: it moves the *audio* route into the
     * telephony regime without the radio state.
     *
     * It is an approximation, not a substitute. A real call additionally holds RIL
     * voice-channel state and may apply carrier/IMS audio policy that setMode does
     * not. So:
     *   - setMode accepted  -> strong evidence about the audio path, still worth an
     *                          eSIM run for the real answer.
     *   - setMode rejected  -> the regime is unreachable without privileges, and the
     *                          carrier-call question cannot be answered on this device
     *                          at all.
     *
     * Either outcome is a result. Reports the observed mode back rather than assuming.
     */
    private fun forceInCallMode() {
        val am = getSystemService(Context.AUDIO_SERVICE) as? android.media.AudioManager
        if (am == null) {
            statusText.text = "no AudioManager"
            return
        }
        val before = am.mode
        val requested = android.media.AudioManager.MODE_IN_CALL
        val applied = try {
            @Suppress("DEPRECATION")
            am.setMode(requested)
            true
        } catch (e: Exception) {
            Log.i(TAG, "[CALLPROBE] setMode threw ${e.javaClass.simpleName}: ${e.message}")
            false
        }
        // setMode is advisory; the authoritative read is getMode() after the fact.
        val after = am.mode
        Log.i(
            TAG,
            "[CALLPROBE] setMode(MODE_IN_CALL) requested=$requested applied=$applied " +
                "mode_before=$before mode_after=$after " +
                "TAKEN=${after == android.media.AudioManager.MODE_IN_CALL}"
        )
        statusText.text = if (after == android.media.AudioManager.MODE_IN_CALL) {
            "MODE_IN_CALL active — measuring now"
        } else {
            "setMode did not take (mode=$after) — platform restricts it"
        }
        measureAll()
    }

    /** Return the audio subsystem to normal so later measurements are not skewed. */
    private fun restoreNormalMode() {
        val am = getSystemService(Context.AUDIO_SERVICE) as? android.media.AudioManager
        if (am == null) return
        @Suppress("DEPRECATION")
        am.setMode(android.media.AudioManager.MODE_NORMAL)
        Log.i(TAG, "[CALLPROBE] mode restored to ${am.mode}")
        statusText.text = "mode restored to NORMAL"
    }

    /**
     * Hold the microphone open on a background thread for [HOLD_MS], logging its own
     * RMS as it goes.
     *
     * This stands in for Omi's own capture engine. The question it answers is not
     * about telephony — it is whether Android permits a SECOND app (the agent's local
     * brain) to read the microphone while the first (Omi's capture pipeline) already
     * holds it. If it does not, then an on-device brain cannot listen on a device
     * running Omi, and that constrains every local-inference design regardless of how
     * the carrier-call question resolves.
     *
     * Held in-process rather than in a second app purely so it is reproducible from
     * adb alone; the platform behaviour is the same, because the constraint is one
     * RECORD_AUDIO consumer, not process boundaries.
     */
    private fun holdMic() {
        val am = getSystemService(Context.AUDIO_SERVICE) as? android.media.AudioManager
        // Ask for the mic the way a call-like app would. If this request is what
        // starves a competitor, it is part of what we are testing.
        @Suppress("DEPRECATION")
        am?.requestAudioFocus(
            null, android.media.AudioManager.STREAM_MUSIC, android.media.AudioManager.AUDIOFOCUS_GAIN_TRANSIENT
        )

        Thread {
            val minBuf = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuf <= 0) {
                Log.w(TAG, "[CALLPROBE] holder unsupported minBuf=$minBuf")
                return@Thread
            }
            var rec: AudioRecord? = null
            try {
                rec = AudioRecord(
                    MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    maxOf(minBuf * 4, SAMPLE_RATE / 4)
                )
                if (rec.state != AudioRecord.STATE_INITIALIZED) {
                    Log.w(TAG, "[CALLPROBE] holder not_initialized state=${rec.state}")
                    return@Thread
                }
                rec.startRecording()
                val buf = ShortArray(maxOf(minBuf * 4, SAMPLE_RATE / 4) / 2)
                var peak = 0.0
                var sumSq = 0.0
                var count = 0L
                val deadline = SystemClock.elapsedRealtime() + HOLD_MS
                Log.i(TAG, "[CALLPROBE] === HOLDER START (stands in for Omi capture) ===")
                while (SystemClock.elapsedRealtime() < deadline) {
                    val n = rec.read(buf, 0, buf.size)
                    if (n <= 0) continue
                    for (i in 0 until n) {
                        val v = buf[i].toDouble()
                        sumSq += v * v
                        count++
                        val a = abs(v)
                        if (a > peak) peak = a
                    }
                }
                val rms = if (count == 0L) 0.0 else sqrt(sumSq / count)
                Log.i(TAG, "[CALLPROBE] === HOLDER END samples=$count rms=${"%.1f".format(rms)} peak=${"%.0f".format(peak)} ===")
            } catch (e: Exception) {
                Log.i(TAG, "[CALLPROBE] holder FAILED ${e.javaClass.simpleName}: ${e.message}")
            } finally {
                try {
                    rec?.stop()
                } catch (_: Exception) {
                }
                rec?.release()
            }
        }.also { it.isDaemon = true; it.start() }
        statusText.text = "holding mic for ${HOLD_MS / 1000}s — run MEASURE now"
    }

    /**
     * Continuous MIC monitor for [MONITOR_MS], one RMS sample per second, with the
     * call state logged alongside.
     *
     * The decisive experiment is a TRANSITION, not two snapshots: RMS before the call
     * connects, at the moment it connects, and after. A one-shot measurement can be
     * misread; a time series that drops exactly when telephony takes the audio route
     * is not.
     *
     * Background caveat, stated because it can confound this badly: Android 9+
     * restricts mic access for backgrounded apps, and dialling pushes this activity
     * into the background. If the series flatlines at zero the instant the activity
     * goes to background rather than at the moment the call connects, that is
     * background restriction, NOT telephony. The call-state lines let the two be told
     * apart, and that is the whole reason they are interleaved here.
     */
    private fun monitor() {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            statusText.text = "MIC unsupported"
            return
        }
        val bufBytes = maxOf(minBuf * 4, SAMPLE_RATE / 4)
        var rec: AudioRecord? = null
        Thread {
            try {
                rec = AudioRecord(
                    MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    bufBytes
                )
                if (rec.state != AudioRecord.STATE_INITIALIZED) {
                    Log.w(TAG, "[CALLPROBE] monitor not_initialized state=${rec.state}")
                    return@Thread
                }
                rec.startRecording()
                val buf = ShortArray(bufBytes / 2)
                var second = 0
                var peak = 0.0
                var sumSq = 0.0
                var count = 0L
                val deadline = SystemClock.elapsedRealtime() + MONITOR_MS
                Log.i(
                    TAG,
                    "[CALLPROBE] === MONITOR START ${MONITOR_MS / 1000}s " +
                        "callActive=${callActive.get()} — dial now ==="
                )
                while (SystemClock.elapsedRealtime() < deadline) {
                    val n = rec.read(buf, 0, buf.size)
                    if (n > 0) {
                        for (i in 0 until n) {
                            val v = buf[i].toDouble()
                            sumSq += v * v
                            count++
                            val a = abs(v)
                            if (a > peak) peak = a
                        }
                    }
                    if (SystemClock.elapsedRealtime() % 1000L < 250L) {
                        second++
                        val rms = if (count == 0L) 0.0 else sqrt(sumSq / count)
                        Log.i(
                            TAG,
                            "[CALLPROBE] MON t=${second}s rms=${"%.1f".format(rms)} " +
                                "peak=${"%.0f".format(peak)} callActive=${callActive.get()}"
                        )
                        peak = 0.0
                        sumSq = 0.0
                        count = 0L
                    }
                }
                Log.i(TAG, "[CALLPROBE] === MONITOR END ===")
            } catch (e: Exception) {
                Log.i(TAG, "[CALLPROBE] monitor FAILED ${e.javaClass.simpleName}: ${e.message}")
            } finally {
                try {
                    rec?.stop()
                } catch (_: Exception) {
                }
                rec?.release()
            }
        }.also { it.isDaemon = true; it.start() }
        statusText.text = "monitoring ${MONITOR_MS / 1000}s — dial now"
    }

    private fun runAll() {
        requestPermissions()
        selfTest()
        measureAll()
    }

    override fun onDestroy() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && telemetry != null) {
            try {
                telemetry!!.unregisterTelephonyCallback(callStateCallback)
            } catch (_: Exception) {
            }
        }
        super.onDestroy()
    }

    private fun abs(v: Double): Double = if (v < 0) -v else v
}
