#!/usr/bin/env bash
set -x
set -e

if [ -z "$TASK" ]; then
  TASK="musique"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}/.."
DATA_DIR="${DIR}/../Datasets/${TASK}"

OUTPUT_DIR="${DIR}/checkpoint/${TASK}/"
mkdir -p "${OUTPUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# sentence-transformers/all-mpnet-base-v2
python -u "${DIR}/main.py" \
  --task "musique" \
  --model-dir "${OUTPUT_DIR}" \
  --pretrained-model "sentence-transformers/all-mpnet-base-v2" \
  --lr 1e-6 \
  --train-path "${DATA_DIR}/preprocess/train.json" \
  --valid-path "${DATA_DIR}/preprocess/valid.json" \
  --batch-size 40 \
  --print-freq 100 \
  --finetune-t \
  --epochs 8 \
  --workers 0 \
  --max-to-keep 3

echo "Training Complete. Checkpoints saved in ${OUTPUT_DIR}"

