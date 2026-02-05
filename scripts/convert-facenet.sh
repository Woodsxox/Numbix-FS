#!/usr/bin/env bash
# Convert a FaceNet Keras .h5 model to TensorFlow.js Layers format and write
# model.json + weight shards into public/models/facenet/.
#
# Usage:
#   ./scripts/convert-facenet.sh /path/to/facenet_keras.h5
#
# Prereq: pip install tensorflowjs
#   (or: pip install tensorflowjs[wizard])

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_ROOT/public/models/facenet"

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/facenet_keras.h5"
  echo ""
  echo "Example: $0 \"$PROJECT_ROOT/archive 2/facenet_keras.h5\""
  exit 1
fi

H5_PATH="$1"
if [ ! -f "$H5_PATH" ]; then
  echo "Error: Not a file: $H5_PATH"
  exit 1
fi

if ! command -v tensorflowjs_converter &>/dev/null; then
  echo "Error: tensorflowjs_converter not found."
  echo "Install with: python3 -m pip install tensorflowjs"
  exit 1
fi

mkdir -p "$OUT_DIR"

# Some tensorflowjs_converter installs choke on paths with spaces; use a copy if needed
INPUT_FOR_CONVERTER="$H5_PATH"
if [[ "$H5_PATH" == *" "* ]]; then
  TMP_H5="/tmp/facenet_keras_$$.h5"
  cp "$H5_PATH" "$TMP_H5"
  INPUT_FOR_CONVERTER="$TMP_H5"
  trap "rm -f $TMP_H5" EXIT
fi

echo "Converting $H5_PATH -> $OUT_DIR"
tensorflowjs_converter \
  --input_format=keras \
  --output_format=tfjs_layers_model \
  "$INPUT_FOR_CONVERTER" \
  "$OUT_DIR"

echo "Done. Check $OUT_DIR for model.json and *.bin files."
