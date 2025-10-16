#!/usr/bin/env bash
set -x
set -e

if [ -z "$TASK" ]; then
  TASK="2wikimultihop"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}/.."

DATA_DIR="${DIR}/../Datasets"

OUTPUT_DIR="${DIR}/checkpoint/${TASK}"
mkdir -p "${OUTPUT_DIR}"

python -u "${DIR}/eval_ircot.py" \
  --checkpoint-path "${OUTPUT_DIR}/model_last.mdl" \
  --test-path "${DATA_DIR}/Test/test_2wiki.json" \
  --results-path "${DIR}/results/retrieved_ircot_${TASK}.json" \
  --pretrained-model "sentence-transformers/all-mpnet-base-v2"

echo "Evaluation complete. Results saved in ${DIR}/results/retrieved_ircot_${TASK}.json"

