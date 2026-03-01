#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="/mnt/disk_3/guanxing/kiseki/.venv/bin/python"

NGPU=1
MASTER_PORT=29503
LOG_RANK=0

MODEL_PATH="/mnt/disk_3/guanxing/dreamzero/MODEL/DreamZero-DROID"
WAN_PRETRAINED_DIR="/mnt/disk_3/guanxing/dreamzero/MODEL/Wan2.1-I2V-14B-480P"
PRESET="franka"
PORT=8000
INDEX=0
SEED=42

NUM_CHUNKS=1
FRAMES_PER_CHUNK=4
NUM_INFERENCE_STEPS=1
NUM_DIT_STEPS=1
NUM_LAYERS=1
CFG_SCALE=1.0

OUTPUT_ROOT="outputs"
RUN_NAME="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
DEBUG_TENSORS_DIR="${OUTPUT_DIR}/debug_tensors"

export DREAMZERO_SKIP_COMPONENT_LOADING="false"
export DREAMZERO_WAN_PRETRAINED_DIR="${WAN_PRETRAINED_DIR}"
export DREAMZERO_DIT_NUM_LAYERS="${NUM_LAYERS}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

unset PYTHONHOME
unset PYTHONPATH

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node="${NGPU}"
  --local-ranks-filter="${LOG_RANK}"
  --master_port "${MASTER_PORT}"
  --tee 3
  local_infer_example_png_AR.py
  --model-path "${MODEL_PATH}"
  --preset "${PRESET}"
  --port "${PORT}"
  --seed "${SEED}"
  --index "${INDEX}"
  --num-chunks "${NUM_CHUNKS}"
  --frames-per-chunk "${FRAMES_PER_CHUNK}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --num-dit-steps "${NUM_DIT_STEPS}"
  --num-layers "${NUM_LAYERS}"
  --cfg-scale "${CFG_SCALE}"
  --enable-deterministic
  --enable-dit-cache
  --save-debug-tensors
  --debug-rank0-only
  --skip-initial-single
  --debug-tensors-dir "${DEBUG_TENSORS_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

echo "Output dir: ${OUTPUT_DIR}"
echo "Debug dir: ${DEBUG_TENSORS_DIR}"
echo "Skip initial single: true"
echo "Python: ${PYTHON_BIN}"

"${cmd[@]}"
