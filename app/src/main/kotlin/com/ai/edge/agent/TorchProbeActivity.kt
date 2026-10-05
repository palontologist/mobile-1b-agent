/**
 * TorchProbeActivity — a real, runnable on-device test of torch control.
 *
 * Purpose is not UX. It exists to emit a reproducible evidence trail on the SM-A14:
 * every command logs a machine-parseable `[PROBE]` line to logcat with latency, and
 * the on-screen text mirrors it so a screenshot matches the log. That is the artifact
 * upstream threads (LiteRT-LM #2421, #2966, LiteRT #6889) actually ask for and nobody
 * on a 3.5GB budget device has supplied.
 *
 * UI is built programmatically so the module needs no layout files, no AppCompat,
 * no Material dependency, and no resource-resolution failures to debug.
 */
package com.ai.edge.agent

import android.app.Activity
import android.content.Context
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class TorchProbeActivity : Activity() {

    private lateinit var statusText: TextView
    private var commandCount = 0
    private val latenciesMs = mutableListOf<Long>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }

        val capabilities = TextView(this).apply {
            textSize = 14f
            text = getString(
                R.string.probe_capabilities,
                android.os.Build.MANUFACTURER,
                android.os.Build.MODEL,
                android.os.Build.VERSION.SDK_INT,
                TorchManager.isAvailable(this@TorchProbeActivity).toString()
            )
        }

        statusText = TextView(this).apply {
            textSize = 16f
            setPadding(0, 32, 0, 32)
            text = getString(R.string.probe_ready)
        }

        root.addView(capabilities)
        root.addView(statusText)
        root.addView(Button(this).apply {
            text = getString(R.string.btn_enable)
            setOnClickListener { probe("enable") { TorchManager.enable(it) } }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.btn_disable)
            setOnClickListener { probe("disable") { TorchManager.disable(it) } }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.btn_toggle)
            setOnClickListener { probe("toggle") { TorchManager.toggle(it) } }
        })
        root.addView(Button(this).apply {
            text = getString(R.string.btn_hammer)
            setOnClickListener { hammer() }
        })

        setContentView(ScrollView(this).apply { addView(root) })

        Log.i(TAG, "[PROBE] activity created available=${TorchManager.isAvailable(this)}")
    }

    /**
     * Repeatedly flips the torch without settling time. Contending for the flash is
     * how the camera service actually breaks on low-RAM devices, so this is the case
     * worth measuring rather than the single-press happy path.
     */
    private fun hammer() {
        Log.i(TAG, "[PROBE] hammer start n=20")
        var failures = 0
        repeat(20) { i ->
            val s = TorchManager.toggle(this)
            if (s is TorchStatus.Failed || s is TorchStatus.CameraBusy) failures++
            Log.i(TAG, "[PROBE] hammer i=$i state=${TorchManager.isTorchOn} status=${s.describe()}")
        }
        Log.i(TAG, "[PROBE] hammer done failures=$failures/20")
        statusText.text = getString(R.string.probe_hammer_done, failures)
    }

    private fun probe(label: String, block: (Context) -> TorchStatus) {
        commandCount++
        val t0 = SystemClock.elapsedRealtimeNanos()
        val status = block(this)
        val us = (SystemClock.elapsedRealtimeNanos() - t0) / 1000
        latenciesMs.add(us / 1000)
        Log.i(
            TAG,
            "[PROBE] cmd=$label n=$commandCount result=${status.describe()} " +
                "state=${TorchManager.isTorchOn} latency_us=$us"
        )
        statusText.text = getString(
            R.string.probe_result,
            label,
            status.describe(),
            us,
            TorchManager.isTorchOn.toString(),
            commandCount,
            latenciesMs.max(),
            latenciesMs.average().toInt()
        )
    }

    companion object {
        private const val TAG = "TorchProbe"
    }
}
