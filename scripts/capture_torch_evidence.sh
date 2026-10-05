#!/usr/bin/env bash
# Capture a reproducible evidence bundle from a connected device.
#
# This produces the artifact that LiteRT-LM #2421 / #2966 and LiteRT #6889 ask for and
# that this repo previously had none of: timestamped logcat + memory + thermal state
# around a controlled burst of torch commands.
#
# Usage:  scripts/capture_torch_evidence.sh [serial]
# The serial is read from `adb devices` (argument 2), never hardcoded, so the script
# and the repository do not carry a device identifier.
set -euo pipefail

SERIAL="${1:-}"
OUT="measurements/$(date -u +%Y%m%dT%H%M%SZ)"
ADB=(adb)
[ -n "$SERIAL" ] && ADB+=(-s "$SERIAL")

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not on PATH. Install platform-tools or set ANDROID_HOME." >&2
  exit 1
fi

mkdir -p "$OUT"

echo "== device identity"
"${ADB[@]}" shell getprop ro.product.manufacturer > "$OUT/props.txt"
"${ADB[@]}" shell getprop ro.product.model >> "$OUT/props.txt"
"${ADB[@]}" shell getprop ro.build.version.sdk >> "$OUT/props.txt"
"${ADB[@]}" shell getprop ro.boot.hardware >> "$OUT/props.txt"

echo "== free memory before"
"${ADB[@]}" shell dumpsys meminfo > "$OUT/meminfo_before.txt"

echo "== logcat capture (background)"
"${ADB[@]}" logcat -v epoch -c || true
"${ADB[@]}" logcat -v epoch TorchProbe:V ActivityManager:W *:S > "$OUT/logcat.txt" 2>&1 &
LOGCAT_PID=$!
cleanup() { kill "$LOGCAT_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "== launching probe"
"${ADB[@]}" shell am start -n com.ai.edge.agent/.TorchProbeActivity
sleep 3

echo "== driving the UI is manual by design (the hammer button); waiting 25s"
sleep 25

echo "== process memory during"
PID="$("${ADB[@]}" shell pidof com.ai.edge.agent | tr -d '\r' || true)"
if [ -n "$PID" ]; then
  "${ADB[@]}" shell dumpsys meminfo "$PID" > "$OUT/meminfo_pid.txt"
  "${ADB[@]}" shell dumpsys gfxinfo com.ai.edge.agent > "$OUT/gfxinfo.txt" || true
fi

echo "== thermal + battery"
"${ADB[@]}" shell dumpsys thermalservice > "$OUT/thermal.txt" || true
"${ADB[@]}" shell dumpsys battery > "$OUT/battery.txt" || true

cleanup
trap - EXIT

echo
echo "bundle written to $OUT"
echo "grep probe lines:  grep '\\[PROBE\\]' $OUT/logcat.txt"
echo
echo "Before publishing: strip the serial from any pasted output, and confirm no"
echo "personal calendar/contact text appears in logcat (AgentLoop logs prompts)."
