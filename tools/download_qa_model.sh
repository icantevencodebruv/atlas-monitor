#!/bin/bash
set -euo pipefail
APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$APP_ROOT/models"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${MODEL_FILE}"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Model already present: $MODEL_DIR/$MODEL_FILE"
  exit 0
fi

echo "Downloading $MODEL_FILE..."
curl -L --progress-bar -o "$MODEL_DIR/$MODEL_FILE" "$URL"
echo "Done: $MODEL_DIR/$MODEL_FILE"
