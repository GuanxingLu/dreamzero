#!/usr/bin/env bash

# set -euo pipefail
# set -x

# umask 007

NGPU=${NGPU:-1}
MASTER_PORT=${MASTER_PORT:-29503}
LOG_RANK=${LOG_RANK:-0}

# MODEL_PATH=${MODEL_PATH:-""}
MODEL_PATH="/mnt/disk_3/guanxing/dreamzero/MODEL/DreamZero-DROID"
PRESET=${PRESET:-franka}
PROMPT=${PROMPT:-""}
PORT=${PORT:-8000}
INDEX=${INDEX:-0}

SEED=${SEED:-42}
ENABLE_DETERMINISTIC=${ENABLE_DETERMINISTIC:-true}
ENABLE_DIT_CACHE=${ENABLE_DIT_CACHE:-true}
SAVE_DEBUG_TENSORS=${SAVE_DEBUG_TENSORS:-true}
DEBUG_RANK0_ONLY=${DEBUG_RANK0_ONLY:-true}
DEBUG_TENSORS_DIR=${DEBUG_TENSORS_DIR:-""}
OUTPUT_DIR=${OUTPUT_DIR:-""}

NUM_CHUNKS=${NUM_CHUNKS:-1}
FRAMES_PER_CHUNK=${FRAMES_PER_CHUNK:-4}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-1}
NUM_DIT_STEPS=${NUM_DIT_STEPS:-1}
CFG_SCALE=${CFG_SCALE:-1.0}

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required."
  echo "Example: MODEL_PATH=/path/to/checkpoint bash scripts/run_launch_dreamzero_local_png_sync.sh"
  exit 1
fi

to_bool_flag() {
  local key="$1"
  local value="${2,,}"
  if [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" ]]; then
    echo "--${key}"
  else
    echo "--no-${key}"
  fi
}

det_flag="$(to_bool_flag "enable-deterministic" "${ENABLE_DETERMINISTIC}")"
dit_cache_flag="$(to_bool_flag "enable-dit-cache" "${ENABLE_DIT_CACHE}")"
debug_flag="$(to_bool_flag "save-debug-tensors" "${SAVE_DEBUG_TENSORS}")"
debug_rank0_only_flag="$(to_bool_flag "debug-rank0-only" "${DEBUG_RANK0_ONLY}")"

cmd=(
  python -m torch.distributed.run
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
  --cfg-scale "${CFG_SCALE}"
  "${det_flag}"
  "${dit_cache_flag}"
  "${debug_flag}"
  "${debug_rank0_only_flag}"
)

if [[ -n "${PROMPT}" ]]; then
  cmd+=(--prompt "${PROMPT}")
fi

if [[ -n "${DEBUG_TENSORS_DIR}" ]]; then
  cmd+=(--debug-tensors-dir "${DEBUG_TENSORS_DIR}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

cmd+=("$@")

"${cmd[@]}"
