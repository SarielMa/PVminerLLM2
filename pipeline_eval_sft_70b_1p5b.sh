#!/usr/bin/env bash
set -euo pipefail

# =========================
# GLOBAL CONFIG
# =========================
REPO_ROOT="$(readlink -f "$(dirname "$0")")"
SFT_EVAL_OUT_ROOT="${SFT_EVAL_OUT_ROOT:-${PIPELINE_OUT_ROOT:-$(readlink -f "${REPO_ROOT}/sft_eval_outputs")}}"

# FinBen task definitions for lm_eval.
FINBEN_TASKS_PATH="${FINBEN_TASKS_PATH:-/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner}"

EPOCHS=3   # must match SFT

# One knob to rule them all
TP="${TP:-2}"
NUM_GPUS="${NUM_GPUS:-${TP}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TP}}"

MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.90

# =========================
# LOCAL SFT MODELS
# =========================
MODELS=(
  "${REPO_ROOT}/PVminerLLM_70b_llama3.3_instruct"
  "${REPO_ROOT}/PVminerLLM_qwen2.5_1.5b_instruct"
)

mkdir -p "${SFT_EVAL_OUT_ROOT}"
SFT_EVAL_OUT_ROOT="$(readlink -f "${SFT_EVAL_OUT_ROOT}")"

if [[ ! -d "${FINBEN_TASKS_PATH}" ]]; then
  echo "WARNING: FINBEN_TASKS_PATH not found, lm_eval will be skipped:"
  echo "  ${FINBEN_TASKS_PATH}"
fi

# =========================
# MAIN LOOP
# =========================
for MODEL in "${MODELS[@]}"; do

  MODEL_TAG="$(basename "${MODEL}")"

  # --------------------------------------------------
  # Locate local SFT model
  # --------------------------------------------------
  SFT_MODEL="$(readlink -f "${MODEL}")"

  if [[ ! -f "${SFT_MODEL}/config.json" ]]; then
    echo "ERROR: local SFT model not found:"
    echo "  ${SFT_MODEL}"
    echo "Skipping ${MODEL}"
    continue
  fi

  # --------------------------------------------------
  # Output folders (one per SFT model)
  # --------------------------------------------------
  OUT_TAG="${MODEL_TAG}_epoch${EPOCHS}_localSft_directEval"
  OUT_ROOT="${SFT_EVAL_OUT_ROOT}/${OUT_TAG}"

  EVAL_DIR="${OUT_ROOT}/lm_eval_results"
  FINBEN_OUT="${EVAL_DIR}/PvExtraction_full"

  mkdir -p "${EVAL_DIR}" "${FINBEN_OUT}"
  FINBEN_OUT="$(readlink -f "${FINBEN_OUT}")"
  case "${FINBEN_OUT}" in
    /*) ;;
    *)
      echo "ERROR: FINBEN_OUT is not an absolute path: ${FINBEN_OUT}" >&2
      exit 1
      ;;
  esac

  echo "============================================================"
  echo "MODEL      : ${MODEL_TAG}"
  echo "SFT_MODEL  : ${SFT_MODEL}"
  echo "OUT_ROOT   : ${OUT_ROOT}"
  echo "FINBEN_OUT : ${FINBEN_OUT}"
  echo "TP/GPUS    : TP=${TP} NUM_GPUS=${NUM_GPUS}"
  echo "============================================================"

  # =========================
  # Eval SFT model directly (lm_eval + vLLM)
  # =========================
  if [[ -d "${FINBEN_TASKS_PATH}" ]]; then
    lm_eval --model vllm \
      --model_args "pretrained=${SFT_MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
      --tasks PvExtraction_full \
      --num_fewshot 0 \
      --batch_size auto \
      --output_path "${FINBEN_OUT}" \
      --log_samples \
      --apply_chat_template \
      --include_path "${FINBEN_TASKS_PATH}"
  else
    echo "WARNING: FINBEN_TASKS_PATH not found, skipping lm_eval:"
    echo "  ${FINBEN_TASKS_PATH}"
  fi

  echo "DONE: ${MODEL_TAG}"
done

echo
echo "All direct SFT eval runs finished. Outputs under:"
echo "  ${SFT_EVAL_OUT_ROOT}"
