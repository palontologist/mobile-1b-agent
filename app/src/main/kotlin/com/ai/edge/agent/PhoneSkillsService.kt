package com.ai.edge.agent

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityService.GestureResultCallback
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

/**
 * Phone Assistant skills — the on-device equivalent of phone-harness's helpers.
 *
 * WHY AN ACCESSIBILITY SERVICE
 *
 * phone-harness drives Android over `adb` from the host, so it needs nothing on the
 * device. On-device there is no public API for "read this other app's screen and tap
 * it". AccessibilityService is the only supported mechanism, and it is a powerful
 * permission: it can read any screen the user sees and dispatch gestures. That is
 * exactly what a skills layer needs and exactly why Play Store policy requires a
 * real accessibility justification for it. If this ever ships publicly, that review
 * is the gate — not the code.
 *
 * WHAT THIS MIRRORS FROM phone-harness
 *
 *   find_text("Weather") -> (x, y)     then  tap(x, y)
 *   open_app("Notes")                   then  type_text(...)
 *   screen() -> [{text, bounds}, ...]
 *
 * The helpers here are deliberately the same shape so a skill written against one is
 * portable to the other. The difference is transport: adb versus AccessibilityService.
 *
 * EVERY ACTION IS LOGGED. Not for telemetry — so a skill can be debugged by reading
 * one stream instead of attaching a debugger to a phone.
 */
class PhoneSkillsService : AccessibilityService() {

    companion object {
        private const val TAG = "PhoneSkills"
        private const val TIMEOUT_FG_MS = 4000L
        @Volatile @JvmStatic var instance: PhoneSkillsService? = null

        /** True once the system has bound us and enabled accessibility. */
        val isReady: Boolean get() = instance != null
    }

    private var skillsSeq = 0
    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "connected — skills are live")
    }

    override fun onDestroy() {
        instance = null
        Log.i(TAG, "destroyed")
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // No-op on purpose. This service acts on command, not on every window
        // change: reacting to events would make skills fire unpredictably. The event
        // types are declared so the framework permits the reads we do do.
    }

    override fun onInterrupt() {
        Log.i(TAG, "interrupted")
    }

    // --- skills -------------------------------------------------------------

    /** `open_app("Notes")` — launch an app by human-readable label or package name. */
    fun openApp(query: String): Boolean {
        val skill = ++skillsSeq
        val packageManager = packageManager
        val launcherIntent = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)

        val candidates = packageManager.queryIntentActivities(launcherIntent, 0)
        val match = candidates.firstOrNull { info ->
            val label = info.loadLabel(packageManager)?.toString().orEmpty()
            label.equals(query, ignoreCase = true) ||
                info.activityInfo.packageName.equals(query, ignoreCase = true) ||
                label.contains(query, ignoreCase = true)
        }
        if (match == null) {
            Log.w(TAG, "open_app skill=$skill query=$query UNMATCHED of ${candidates.size}")
            return false
        }
        val pkg = match.activityInfo.packageName
        val t0 = SystemClock.elapsedRealtime()
        val intent = packageManager.getLaunchIntentForPackage(pkg)?.apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        if (intent == null) {
            Log.w(TAG, "open_app skill=$skill pkg=$pkg NO_LAUNCH_INTENT")
            return false
        }
        return try {
            startActivity(intent)
            val ms = SystemClock.elapsedRealtime() - t0
            Log.i(
                TAG,
                "open_app skill=$skill query=$query pkg=$pkg label=" +
                    "${match.loadLabel(packageManager)} launch_ms=$ms"
            )
            true
        } catch (e: Exception) {
            Log.e(TAG, "open_app skill=$skill pkg=$pkg FAILED ${e.message}")
            false
        }
    }

    /** Every launchable app, so a skill can pick by label rather than guessing a package. */
    fun listApps(): List<Pair<String, String>> {
        val pm = packageManager
        val out = pm.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)
            .mapNotNull { info ->
                val label = info.loadLabel(pm)?.toString()?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                label to info.activityInfo.packageName
            }
            .distinctBy { it.first }
            .sortedBy { it.first.lowercase() }
        Log.i(TAG, "list_apps count=${out.size}")
        return out
    }

    /**
     * `screen()` — the accessibility tree as flat text + bounds.
     *
     * Depth-limited and cycle-guarded: a real tree can be deep enough to blow the
     * stack on some OEM skins, and a cycle would hang the walk.
     */
    fun readScreen(maxNodes: Int = 400, maxDepth: Int = 40): List<ScreenNode> {
        val root = rootInActiveWindow ?: run {
            Log.w(TAG, "screen NO_ACTIVE_WINDOW")
            return emptyList()
        }
        val out = mutableListOf<ScreenNode>()
        val visited = java.util.Collections.newSetFromMap(java.util.IdentityHashMap<AccessibilityNodeInfo, Boolean>())
        walk(root, Rect(), 0, maxNodes, maxDepth, visited, out)
        Log.i(TAG, "screen nodes=${out.size} depth_capped=${maxDepth} node_capped=$maxNodes")
        return out
    }

    private fun walk(
        node: AccessibilityNodeInfo,
        clip: Rect,
        depth: Int,
        maxNodes: Int,
        maxDepth: Int,
        visited: MutableSet<AccessibilityNodeInfo>,
        out: MutableList<ScreenNode>
    ) {
        if (out.size >= maxNodes || depth > maxDepth) return
        if (!visited.add(node)) return

        val bounds = Rect().also { node.getBoundsInScreen(it) }
        val text = buildString {
            append(node.text?.toString().orEmpty())
            node.contentDescription?.toString()?.let { append(it) }
        }.trim()

        if (text.isNotEmpty() && bounds.width() > 0 && bounds.height() > 0) {
            out.add(ScreenNode(text, bounds, node.className?.toString().orEmpty(), depth))
        }

        for (i in 0 until node.childCount) {
            val child = try { node.getChild(i) } catch (_: Exception) { null } ?: continue
            walk(child, clip, depth + 1, maxNodes, maxDepth, visited, out)
            try { child.recycle() } catch (_: Exception) {}
        }
    }

    /**
     * `find_text("Weather")` — locate a label and return a tappable point.
     *
     * Prefers the node's centre, but only when the node is actually clickable or
     * looks like one. An ancestor often carries the text while the clickable child
     * is a zero-sized overlay, and tapping the text can land between two controls.
     */
    fun findText(query: String): Point? {
        val nodes = readScreen()
        val needle = query.lowercase()
        val hit = nodes.firstOrNull { it.text.lowercase().contains(needle) }
        if (hit == null) {
            Log.w(TAG, "find_text query=$query NOT_FOUND (${nodes.size} nodes scanned)")
            return null
        }
        val point = clickableCentre(nodes, hit.bounds)
        Log.i(
            TAG,
            "find_text query=$query match='${hit.text.take(40)}' class=${hit.className} " +
                "bounds=${hit.bounds} -> point=$point"
        )
        return point
    }

    /** Walk up from the matched bounds to find an ancestor marked clickable. */
    private fun clickableCentre(nodes: List<ScreenNode>, target: Rect): Point {
        val ancestors = nodes.filter { it.bounds.contains(target) || target.contains(it.bounds) }
            .sortedBy { it.depth }
        val clickable = ancestors.firstOrNull { it.className.contains("Button", true) || it.className.contains("clickable", true) }
        val box = clickable?.bounds ?: target
        return Point(box.centerX(), box.centerY())
    }

    /** `tap(x, y)` — dispatch a gesture. Returns false when the gesture is rejected. */
    fun tap(x: Int, y: Int): Boolean {
        val skill = ++skillsSeq
        val t0 = SystemClock.elapsedRealtime()
        val path = Path().apply { moveTo(x.toFloat(), y.toFloat()) }
        val ok = dispatchGesture(
            GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, 60))
                .build(),
            object : GestureResultCallback() {
                override fun onCompleted(description: GestureDescription?) {
                    Log.i(TAG, "tap skill=$skill at=($x,$y) ok=true ms=${SystemClock.elapsedRealtime() - t0}")
                }

                override fun onCancelled(description: GestureDescription?) {
                    Log.w(TAG, "tap skill=$skill at=($x,$y) CANCELLED ms=${SystemClock.elapsedRealtime() - t0}")
                }
            },
            null
        )
        if (!ok) Log.w(TAG, "tap skill=$skill at=($x,$y) REJECTED")
        return ok
    }

    /** `type_text("hello")` — set text on whatever currently holds focus. */
    fun typeText(text: String): Boolean {
        val focused = rootInActiveWindow?.findFocus(android.view.accessibility.AccessibilityNodeInfo.FOCUS_INPUT)
        if (focused == null) {
            Log.w(TAG, "type_text NOT_IMPLEMENTED no_focused_input len=${text.length}")
            return false
        }
        // The ACTION_SET_TEXT Int constant is for performAction(); the Bundle key is the
        // literal string, not the constant. Easy to conflate, and it does not compile
        // if you try.
        val arg = android.os.Bundle().apply {
            putCharSequence("ACTION_SET_TEXT", text)
        }
        val done = focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arg)
        Log.i(TAG, "type_text len=${text.length} class=${focused.className} ok=$done")
        try { focused.recycle() } catch (_: Exception) {}
        return done
    }

    // --- sequenced skills ---------------------------------------------------
    //
    // Why this exists: `openApp` starts another app, which BACKGROUNDs the assistant
    // activity. So a skill cannot be a sequence of button presses — once the target
    // app is up, there is no button left to press. Sequencing therefore has to live
    // in the service, which survives the switch, and has to WAIT for the new window
    // before reading it.
    //
    // This is also the shape a model emits eventually: a list of steps, each one a
    // helper the model picked, executed here rather than in the UI.

    /** One step of a skill sequence. Mirrors a single phone-harness helper call. */
    sealed interface SkillStep {
        data class OpenApp(val query: String) : SkillStep
        data class ReadScreen(val label: String = "") : SkillStep
        data class FindText(val query: String) : SkillStep
        data class TapAt(val x: Int, val y: Int) : SkillStep
        data class TypeText(val text: String) : SkillStep
        data class Wait(val ms: Long) : SkillStep
    }

    data class StepResult(val step: SkillStep, val ok: Boolean, val detail: String)

    /**
     * Run [steps] in order on a background thread, calling [onDone] on the main
     * thread.
     *
     * Returns immediately. A sequence that opens another app will outlive whatever
     * launched it, which is the entire point — see the comment above.
     */
    fun runSequence(steps: List<SkillStep>, onDone: (List<StepResult>) -> Unit) {
        Thread({
            val results = mutableListOf<StepResult>()
            for (step in steps) {
                results.add(executeStep(step))
            }
            Log.i(TAG, "sequence_done steps=${steps.size} ok=${results.count { it.ok }}")
            mainHandler.post { onDone(results) }
        }, "skills-sequence").apply { isDaemon = true; start() }
    }

    private fun executeStep(step: SkillStep): StepResult = try {
        when (step) {
            is SkillStep.OpenApp -> {
                val pkgBefore = foregroundPackage()
                val ok = openApp(step.query)
                val pkgAfter = waitForForegroundChange(pkgBefore, TIMEOUT_FG_MS)
                StepResult(step, ok, "opened=${step.query} foreground=$pkgAfter")
            }

            is SkillStep.ReadScreen -> {
                val nodes = readScreen()
                val preview = nodes.take(6).joinToString(" | ") { it.text.take(28) }
                StepResult(step, nodes.isNotEmpty(), "nodes=${nodes.size} [$preview]")
            }

            is SkillStep.FindText -> {
                val point = findText(step.query)
                StepResult(step, point != null, if (point != null) "point=(${point.x},${point.y})" else "not found")
            }

            is SkillStep.TapAt -> StepResult(step, tap(step.x, step.y), "tap(${step.x},${step.y})")

            is SkillStep.TypeText -> StepResult(step, typeText(step.text), "type(len=${step.text.length})")

            is SkillStep.Wait -> {
                Thread.sleep(step.ms.coerceIn(0, 10_000))
                StepResult(step, true, "waited ${step.ms}ms")
            }
        }
    } catch (e: Exception) {
        Log.e(TAG, "step_failed $step", e)
        StepResult(step, false, e.javaClass.simpleName + ": " + e.message)
    }

    private fun foregroundPackage(): String? =
        rootInActiveWindow?.let { root ->
            // Walk down for the first node that names a package; the root itself
            // usually reports the service, not the app on screen.
            var node: AccessibilityNodeInfo? = root
            var depth = 0
            while (node != null && depth < 20) {
                node.packageName?.toString()?.takeIf { it.isNotEmpty() && it != packageName }
                    ?.let { return it }
                node = try { node.getChild(0) } catch (_: Exception) { null }
                depth++
            }
            null
        }

    /**
     * Wait until the foreground package differs from [previous].
     *
     * Without this, a read immediately after `openApp` races the window swap and
     * reads OUR OWN still-visible activity — which is exactly the stale-read bug
     * that makes sequenced skills look broken.
     */
    private fun waitForForegroundChange(previous: String?, timeoutMs: Long): String? {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            val now = foregroundPackage()
            if (now != null && now != previous) return now
            Thread.sleep(120)
        }
        return null
    }

    data class ScreenNode(val text: String, val bounds: Rect, val className: String, val depth: Int)
    data class Point(val x: Int, val y: Int)
}