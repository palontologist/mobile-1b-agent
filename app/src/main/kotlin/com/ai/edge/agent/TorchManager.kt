/**
 * TorchManager - flashlight control via android.hardware.camera2.CameraManager.
 *
 * Why this file exists: ActionRegistry referenced a `TorchManager` obtained through
 * `ctx.getSystemService(TorchService::class.java)`, but neither type was ever defined
 * and `getSystemService` is not how you reach the flash. There is no Android API that
 * hands you a "torch service"; the supported path (API 23+) is:
 *
 *   CameraManager.setTorchMode(cameraId, enabled)
 *
 * Notes that matter on a budget Exynos device (SM-A14):
 *  - `setTorchMode` and `getCameraCharacteristics` need **no** CAMERA permission.
 *    Opening a camera session would. That is why a torch can be driven from the
 *    background, and why the manifest stays small.
 *  - The call is synchronous but throws `CameraAccessException`; the common real-world
 *    failure on low-RAM phones is `cameraDeviceInUse`/`inProgress` because the camera
 *    service is contended, so that case gets its own status instead of a generic error.
 *  - Torch state is not reliably readable back. `registerTorchCallback` is the only
 *    authoritative source, so we track the last command we believe succeeded and treat
 *    it as advisory, never as ground truth.
 */
package com.ai.edge.agent

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import android.content.pm.PackageManager

/** Outcome of a torch command, so callers can speak a useful sentence. */
sealed class TorchStatus {
    object TurnedOn : TorchStatus()
    object TurnedOff : TorchStatus()
    object AlreadyOn : TorchStatus()
    object AlreadyOff : TorchStatus()
    data class Unsupported(val reason: String) : TorchStatus()
    data class CameraBusy(val reason: String) : TorchStatus()
    data class Failed(val reason: String) : TorchStatus()
}

object TorchManager {

    private const val TAG = "TorchManager"

    /** Last state we successfully commanded. Advisory only — see class docs. */
    @Volatile
    var isTorchOn: Boolean = false
        private set

    private fun cameraManager(ctx: Context): CameraManager =
        ctx.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private fun hasFlashUnit(ctx: Context): Boolean =
        ctx.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_FLASH)

    /**
     * The id of the rear camera that actually owns a flash unit.
     * Returns null when the device has no usable torch.
     */
    private fun flashCameraId(ctx: Context): String? = try {
        val cm = cameraManager(ctx)
        val ids = cm.cameraIdList
        fun flashOf(id: String) =
            cm.getCameraCharacteristics(id).get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false
        // Prefer a rear camera that owns the flash; fall back to any camera that has one.
        ids.firstOrNull { id ->
            flashOf(id) && cm.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_FRONT
        } ?: ids.firstOrNull { flashOf(it) }
    } catch (e: Exception) {
        Log.w(TAG, "could not enumerate cameras: ${e.message}")
        null
    }

    /** Cheap capability check for onboarding/diagnostics. */
    fun isAvailable(ctx: Context): Boolean = hasFlashUnit(ctx) && flashCameraId(ctx) != null

    fun enable(ctx: Context): TorchStatus = setTorch(ctx, true)

    fun disable(ctx: Context): TorchStatus = setTorch(ctx, false)

    /** Returns the state the caller should treat as current after this call. */
    fun toggle(ctx: Context): TorchStatus = setTorch(ctx, !isTorchOn)

    private fun setTorch(ctx: Context, on: Boolean): TorchStatus {
        if (!hasFlashUnit(ctx)) {
            return TorchStatus.Unsupported("device reports no camera flash unit")
        }
        val id = flashCameraId(ctx)
            ?: return TorchStatus.Unsupported("no camera exposes a flash unit")

        if (isTorchOn == on) {
            return if (on) TorchStatus.AlreadyOn else TorchStatus.AlreadyOff
        }

        return try {
            cameraManager(ctx).setTorchMode(id, on)
            isTorchOn = on
            Log.i(TAG, "torch $on on camera $id")
            if (on) TorchStatus.TurnedOn else TorchStatus.TurnedOff
        } catch (e: android.hardware.camera2.CameraAccessException) {
            val reason = e.message ?: e.javaClass.simpleName
            Log.w(TAG, "setTorchMode($id, $on) failed: $reason")
            // A rejected command means our mirror of reality is stale.
            isTorchOn = !on
            when {
                reason.contains("in use", true) ||
                    reason.contains("CAMERA_IN_USE", true) ||
                    reason.contains("progress", true) -> TorchStatus.CameraBusy(reason)
                else -> TorchStatus.Failed(reason)
            }
        } catch (e: Exception) {
            val reason = e.message ?: e.javaClass.simpleName
            Log.e(TAG, "unexpected torch failure: $reason", e)
            TorchStatus.Failed(reason)
        }
    }

}

/**
 * Top-level so callers outside the object can use it
 * (a member extension is only visible inside TorchManager's scope).
 */
fun TorchStatus.describe(): String = when (this) {
    is TorchStatus.TurnedOn -> "flashlight on"
    is TorchStatus.TurnedOff -> "flashlight off"
    is TorchStatus.AlreadyOn -> "flashlight already on"
    is TorchStatus.AlreadyOff -> "flashlight already off"
    is TorchStatus.Unsupported -> "no flashlight on this device ($reason)"
    is TorchStatus.CameraBusy -> "camera busy, try again ($reason)"
    is TorchStatus.Failed -> "torch failed ($reason)"
}

