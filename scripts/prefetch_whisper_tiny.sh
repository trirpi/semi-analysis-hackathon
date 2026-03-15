#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_DIR="${WHISPER_CACHE_DIR:-$REPO_ROOT/model-cache/whisper}"
mkdir -p "$CACHE_DIR"

URL="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
DEST="$CACHE_DIR/tiny.pt"

if [ -f "$DEST" ]; then
  echo "Checkpoint already exists: $DEST"
  exit 0
fi

echo "Downloading Whisper tiny checkpoint to $DEST"
curl -L --fail --insecure "$URL" -o "$DEST"
echo "Done"
