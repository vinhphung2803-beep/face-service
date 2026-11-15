#!/usr/bin/env bash
set -e

mkdir -p models

MODEL="models/face_recognition_sface_2021dec.onnx"

if [ ! -f "$MODEL" ]; then
    echo "Downloading SFace model..."

    curl -fL \
      "https://github.com/vinhphung2803-beep/face-service/releases/download/v1.0.0/face_recognition_sface_2021dec.onnx" \
      -o "$MODEL"
fi

if [ ! -s "$MODEL" ]; then
    echo "ERROR: SFace model download failed!"
    exit 1
fi

echo "SFace model ready!"
ls -lh models/
