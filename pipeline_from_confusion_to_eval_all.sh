#!/usr/bin/env bash
set -euo pipefail

# =========================
# GLOBAL CONFIG
# =========================
REPO_ROOT="$(readlink -f "$(dirname "$0")")"
PIPELINE_OUT_ROOT="${PIPELINE_OUT_ROOT:-$(readlink -f "${REPO_ROOT}/preference_pipeline_outputs")}"

# Weights and intermediates can live on scratch (PIPELINE_OUT_ROOT), while the
# small eval artifacts stay on the project disk. Point both at the same place
# to restore the old single-tree layout.
RESULTS_ROOT="${RESULTS_ROOT:-$(readlink -f "${REPO_ROOT}/results")}"

# Default to the local HF dataset discovered in the sibling benchmark tree.
DATA_DIR="${DATA_DIR:-$(readlink -f "${REPO_ROOT}/../benckmark/PV_benckmark/split_out/non_test/training")}"


# Label only: this never reaches a trainer. SFT runs 10 epochs
# (sft_epoch10_raw2shot_to_finben_b200.sh) and preference training runs 3
# (train_preference.py Config.num_train_epochs).
SFT_EPOCHS="${SFT_EPOCHS:-10}"

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
# Override with a space-separated list of SFT model dirs, e.g.
# MODELS="${SFT_RESULTS_ROOT}/PVminerLLM_qwen3.5_9b". The dir basename is the tag.
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODELS <<< "${MODELS}"
else
MODELS=(
  "${REPO_ROOT}/PVminerLLM_70b_llama3.3_instruct"
  "${REPO_ROOT}/PVminerLLM_8b_llama3.1_instruct"
  "${REPO_ROOT}/PVminerLLM_3b_llama3.2_instruct"
  "${REPO_ROOT}/PVminerLLM_qwen2.5_1.5b_instruct"
  # Qwen3.5 family: hybrid-reasoning, VLM-wrapped decoders with 3:1
  # linear:full attention. They need chat prompts and the full LoRA target
  # set -- see is_qwen35 below.
  "${SFT_RESULTS_ROOT:-${REPO_ROOT}}/PVminerLLM_qwen3.5_9b"
  "${SFT_RESULTS_ROOT:-${REPO_ROOT}}/PVminerLLM_qwen3.8_27b"
)
fi

# Models whose linear-attention layers expose in_proj_*/out_proj instead of
# q_proj/k_proj/v_proj/o_proj, and whose chat template opens a <think> block
# unless thinking is explicitly disabled.
is_qwen35 () {
  case "$1" in
    *qwen3.5*|*qwen3.8*|*Qwen3.5*|*Qwen3.8*|*qwen3_5*) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "${PIPELINE_OUT_ROOT}" "${RESULTS_ROOT}"

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
  OUT_TAG="${MODEL_TAG}_sft${SFT_EPOCHS}ep_po3ep"
  OUT_ROOT="${PIPELINE_OUT_ROOT}/${OUT_TAG}"

  CONF_DIR="${OUT_ROOT}/confusion"
  PRED_DIR="${OUT_ROOT}/pred"
  PREFERENCE_DATA_DIR="${OUT_ROOT}/preference_data"
  PREFERENCE_RUNS_DIR="${OUT_ROOT}/preference_runs"
  EVAL_DIR="${RESULTS_ROOT}/${OUT_TAG}/eval_results"

  mkdir -p "${CONF_DIR}" "${PRED_DIR}" "${PREFERENCE_DATA_DIR}" "${PREFERENCE_RUNS_DIR}" "${EVAL_DIR}"

  EVAL_DIR="$(readlink -f "${EVAL_DIR}")"

  CODE_CONF_CSV="${CONF_DIR}/code_confusion_summary.csv"
  SUBCODE_CONF_CSV="${CONF_DIR}/subcode_confusion_summary.csv"
  PRED_JSONL="${PRED_DIR}/pred_dump.jsonl"

  RUN_NAME="preference_${OUT_TAG}"
  TRAIN_OUTPUT_DIR="${PREFERENCE_RUNS_DIR}/${RUN_NAME}"
  MERGED_DIR="${PREFERENCE_RUNS_DIR}/${RUN_NAME}-merged"

  # --------------------------------------------------
  # Per-model prompt / LoRA handling
  # --------------------------------------------------
  if is_qwen35 "${MODEL_TAG}"; then
    INFER_EXTRA=(--prompt_mode chat)
    TRAIN_EXTRA=(--lora_target_modules auto)
  else
    INFER_EXTRA=()
    TRAIN_EXTRA=()
  fi

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
    --out_pred_jsonl "${PRED_JSONL}" \
    "${INFER_EXTRA[@]}"

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
    --num_gpus "${NUM_GPUS}" \
    "${TRAIN_EXTRA[@]}"

  # =========================
  # 4) Merge trained adapter
  # =========================
  python merge_lora.py \
    --base "${SFT_MODEL}" \
    --adapter "${TRAIN_OUTPUT_DIR}" \
    --out "${MERGED_DIR}" \
    --dtype bf16

  # =========================
  # 5) Eval (0-shot, chat template, greedy; thinking off)
  # =========================
  python evaluate_pv.py \
    --model "${MERGED_DIR}" \
    --out_dir "${EVAL_DIR}" \
    --tp "${TENSOR_PARALLEL_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}"

  echo "✔ DONE: ${MODEL_TAG}"
done

echo
echo "All preference runs finished. Outputs under:"
echo "  ${PIPELINE_OUT_ROOT}"
