#!/usr/bin/env bash
set -euo pipefail
# Future official acceptance only. Never executes during local Phase F analysis.
# Provide official 100 and 5 image directories and a separately authorized, frozen CSV.
: "${CHECKPOINT:?Set actual relocated candidate checkpoint or F01 bundle path}"
: "${FROZEN_SUBMISSION:?Set final frozen CSV with matching metadata}"
: "${OFFICIAL_TEST_MANIFEST:?Set official manifest for final material verification}"
: "${PICTURES_100:?Set official measured-image directory}"
: "${PICTURES_5:?Set official five-image warmup directory}"
: "${OUTPUT_DIR:?Set a new output directory}"
python -m scripts.accept_t4 "$FROZEN_SUBMISSION" "$PICTURES_100" "$PICTURES_5" \
  --checkpoint "$CHECKPOINT" --test-manifest "$OFFICIAL_TEST_MANIFEST" --output "$OUTPUT_DIR"
# Existing cost.py enforces actual T4, Python3.12 / torch2.10.0 / torchvision0.25.0 /
# CUDA12.8, 100+5, batch1, 3 rounds, <=100ms, <=500MB, consistency and shared Predictor.
