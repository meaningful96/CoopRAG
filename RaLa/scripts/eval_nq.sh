#!/usr/bin/env bash
set -x
set -e

# if TASK is not set, default to hotpotqa
if [ -z "$TASK" ]; then
  TASK="naturalquestions"
fi

# determine script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}/.."

# set dataset, checkpoint, and results paths
DATA_DIR="${DIR}/../Datasets/${TASK}"
CHECKPOINT_PATH="${DIR}/checkpoint/${TASK}/model_last.mdl"
TEST_PATH="${DATA_DIR}/test.json"
RESULTS_DIR="${DIR}/results"
RESULTS_PATH="${RESULTS_DIR}/retrieved_${TASK}.json"

# create results directory if not exists
mkdir -p "${RESULTS_DIR}"

python3 "${DIR}/eval.py" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --test-path "${TEST_PATH}" \
  --results-path "${RESULTS_PATH}" \
  --pretrained-model "sentence-transformers/all-mpnet-base-v2"

