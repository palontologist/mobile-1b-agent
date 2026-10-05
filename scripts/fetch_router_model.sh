#!/usr/bin/env bash
# Fetches the embedding-router model into app/src/main/assets/.
#
# The file is 90 MB and .gitignore excludes *.tflite, so a fresh clone has to fetch
# it. Everything else the app needs (vocab.txt, minilm_tokens.json) IS committed.
#
# Usage:  scripts/fetch_router_model.sh
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/app/src/main/assets"
URL="https://huggingface.co/Bombek1/all-MiniLM-L6-v2-litert/resolve/main/sentence-transformers_all-MiniLM-L6-v2.tflite"
OUT="$DEST/minilm.tflite"

if [[ -f "$OUT" ]]; then
  echo "already present: $OUT"
  exit 0
fi

mkdir -p "$DEST"
echo "downloading MiniLM-L6-v2 (90 MB) -> $OUT"
curl -fsSL -o "$OUT" "$URL"

# vocab.txt is committed, but fetch it too if it is missing so this script works
# from a bare checkout as well.
VOCAB="$DEST/vocab.txt"
if [[ ! -f "$VOCAB" ]]; then
  echo "downloading vocab.txt"
  curl -fsSL -o "$VOCAB" \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"
fi

ls -la "$OUT"
echo
echo "built APK now has the model. Install and launch:"
echo "  ./gradlew :app:assembleDebug"
echo "  adb install -r app/build/outputs/apk/debug/app-debug.apk"
echo "  adb shell am start -n com.ai.edge.agent/.AgentOnPhoneActivity"
echo
echo "or drive it headlessly, no tapping:"
echo "  adb shell am start -n com.ai.edge.agent/.AgentOnPhoneActivity \\"
echo "      --es utterance 'turn on the flashlight'"