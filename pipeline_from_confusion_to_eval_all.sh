#!/usr/bin/env bash
set -euo pipefail

# =========================
# GLOBAL CONFIG
# =========================
REPO_ROOT="$(readlink -f "$(dirname "$0")")"
PIPELINE_OUT_ROOT="${PIPELINE_OUT_ROOT:-$(readlink -f "${REPO_ROOT}/preference_pipeline_outputs")}"

# Default to the local HF dataset discovered in the sibling benchmark tree.
DATA_DIR="${DATA_DIR:-$(readlink -f "${REPO_ROOT}/../benckmark/PV_benckmark/split_out/non_test/training")}"

# Only needed for the final lm_eval step.
FINBEN_TASKS_PATH="${FINBEN_TASKS_PATH:-/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner}"

EPOCHS=3   # must match SFT

# One knob to rule them all
TP="${TP:-2}"
NUM_GPUS="${NUM_GPUS:-${TP}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TP}}"

MAX_TOKENS=8192
TEMPERATURE=0.0
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.90

NEG_PER_SAMPLE=1
SEED=42
PRINT_SAMPLES=3

# =========================
# LOCAL SFT MODELS
# =========================
MODELS=(
  "${REPO_ROOT}/PVminerLLM_70b_llama3.3_instruct"
  "${REPO_ROOT}/PVminerLLM_8b_llama3.1_instruct"
  "${REPO_ROOT}/PVminerLLM_3b_llama3.2_instruct"
  "${REPO_ROOT}/PVminerLLM_qwen2.5_1.5b_instruct"
)

mkdir -p "${PIPELINE_OUT_ROOT}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: DATA_DIR does not exist:"
  echo "  ${DATA_DIR}"
  exit 1
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
  OUT_TAG="${MODEL_TAG}_epoch${EPOCHS}_localSft"
  OUT_ROOT="${PIPELINE_OUT_ROOT}/${OUT_TAG}"

  CONF_DIR="${OUT_ROOT}/confusion"
  PRED_DIR="${OUT_ROOT}/pred"
  PREFERENCE_DATA_DIR="${OUT_ROOT}/preference_data"
  PREFERENCE_RUNS_DIR="${OUT_ROOT}/preference_runs"
  EVAL_DIR="${OUT_ROOT}/lm_eval_results"
  FINBEN_OUT="${EVAL_DIR}/PvExtraction_full"

  mkdir -p "${CONF_DIR}" "${PRED_DIR}" "${PREFERENCE_DATA_DIR}" "${PREFERENCE_RUNS_DIR}" "${EVAL_DIR}" "${FINBEN_OUT}"

  FINBEN_OUT="$(readlink -f "${FINBEN_OUT}")"

  CODE_CONF_CSV="${CONF_DIR}/code_confusion_summary.csv"
  SUBCODE_CONF_CSV="${CONF_DIR}/subcode_confusion_summary.csv"
  PRED_JSONL="${PRED_DIR}/pred_dump.jsonl"

  RUN_NAME="preference_${OUT_TAG}"
  TRAIN_OUTPUT_DIR="${PREFERENCE_RUNS_DIR}/${RUN_NAME}"
  MERGED_DIR="${PREFERENCE_RUNS_DIR}/${RUN_NAME}-merged"

  echo "============================================================"
  echo "MODEL      : ${MODEL_TAG}"
  echo "SFT_MODEL  : ${SFT_MODEL}"
  echo "DATA_DIR   : ${DATA_DIR}"
  echo "OUT_ROOT   : ${OUT_ROOT}"
  echo "TP/GPUS    : TP=${TP} NUM_GPUS=${NUM_GPUS}"
  echo "============================================================"

  # =========================
  # 1) Infer + confusion
  # =========================
  python infer_vllm_and_confusion.py \
    --model "${SFT_MODEL}" \
    --data  "${DATA_DIR}" \
    --out_code_csv "${CODE_CONF_CSV}" \
    --out_subcode_csv "${SUBCODE_CONF_CSV}" \
    --tp "${TP}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --out_pred_jsonl "${PRED_JSONL}"

  # =========================
  # 2) Prepare preference data
  # =========================
  python prepare_preference_data.py \
    --input_dir "${DATA_DIR}" \
    --output_dir "${PREFERENCE_DATA_DIR}" \
    --code_confusion_file "${CODE_CONF_CSV}" \
    --subcode_confusion_file "${SUBCODE_CONF_CSV}" \
    --negatives_per_sample "${NEG_PER_SAMPLE}" \
    --seed "${SEED}" \
    --print_samples "${PRINT_SAMPLES}"

  # =========================
  # 3) Train preference model (LoRA)
  # =========================
  python train_preference.py \
    --model_name "${SFT_MODEL}" \
    --train_data_path "${PREFERENCE_DATA_DIR}" \
    --valid_data_path "${PREFERENCE_DATA_DIR}" \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
    --num_gpus "${NUM_GPUS}"

  # =========================
  # 4) Merge trained adapter
  # =========================
  python merge_lora.py \
    --base "${SFT_MODEL}" \
    --adapter "${TRAIN_OUTPUT_DIR}" \
    --out "${MERGED_DIR}" \
    --dtype bf16

  # =========================
  # 5) Eval (lm_eval + vLLM)
  # =========================
  if [[ -d "${FINBEN_TASKS_PATH}" ]]; then
    lm_eval --model vllm \
      --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
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

  echo "✔ DONE: ${MODEL_TAG}"
done

echo
echo "All preference runs finished. Outputs under:"
echo "  ${PIPELINE_OUT_ROOT}"
