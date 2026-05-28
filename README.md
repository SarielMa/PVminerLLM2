# PVminerLLM2

Code for the paper **"PVminerLLM2: Improving Structured Extraction of Patient Voice via Preference Optimization"**.

This repository contains the training and evaluation pipeline used to improve structured extraction of patient voice annotations with preference optimization. The code focuses on three linked stages:

1. Run model inference and build confusion summaries from prediction errors.
2. Convert those confusions into targeted chosen/rejected preference pairs.
3. Train a LoRA-based preference model and optionally merge the adapter for evaluation.

## Released Models

Trained models are available on Hugging Face:

- `lm2445/PVminerLLM2_1.5B`
- `lm2445/PVminerLLM2_3B`
- `lm2445/PVminerLLM2_8B`
- `lm2445/PVminerLLM2_70B`

## Repository Structure

- [infer_vllm_and_confusion.py](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/infer_vllm_and_confusion.py): runs vLLM inference on a Hugging Face dataset saved with `load_from_disk`, parses structured outputs, and writes code/sub-code confusion CSVs.
- [prepare_preference_data.py](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/prepare_preference_data.py): builds targeted preference pairs using observed confusion patterns.
- [train_preference.py](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/train_preference.py): trains a token-weighted preference objective with LoRA adapters.
- [merge_lora.py](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/merge_lora.py): merges a trained LoRA adapter into the base model.
- [pv_utils.py](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/pv_utils.py): utilities for parsing structured outputs and computing code, sub-code, and span metrics.
- [pipeline_from_confusion_to_eval_all.sh](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/pipeline_from_confusion_to_eval_all.sh): end-to-end pipeline across the local SFT models in this repository.
- [apply_server.sh](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/apply_server.sh): example SLURM submission script for running the pipeline on GPU nodes.

## Expected Data Format

The main scripts expect a Hugging Face dataset directory that can be loaded with `datasets.load_from_disk(...)`.

The inference script expects:

- `query`: input prompt
- `answer`: gold structured output

The structured output format is JSON-like and centers on a `results` list with entries such as:

```json
{
  "results": [
    {
      "Code": "PartnershipPatient",
      "Sub-code": "statePreferences",
      "Span": "example text span"
    }
  ]
}
```

## Environment

An example Conda environment is provided in [environment.yml](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/environment.yml).

```bash
conda env create -f environment.yml
conda activate finben_vllm3
```

Core dependencies used by the pipeline include `transformers`, `datasets`, `peft`, `torch`, `vllm`, `scikit-learn`, and `lm-eval`.

## Minimal Workflow

### 1. Generate confusion summaries

```bash
python infer_vllm_and_confusion.py \
  --model /path/to/base_or_sft_model \
  --data /path/to/hf_dataset \
  --out_code_csv outputs/code_confusion_summary.csv \
  --out_subcode_csv outputs/subcode_confusion_summary.csv \
  --out_pred_jsonl outputs/pred_dump.jsonl \
  --tp 1
```

### 2. Build preference data

```bash
python prepare_preference_data.py \
  --input_dir /path/to/hf_dataset \
  --output_dir outputs/preference_data \
  --code_confusion_file outputs/code_confusion_summary.csv \
  --subcode_confusion_file outputs/subcode_confusion_summary.csv \
  --negatives_per_sample 1 \
  --seed 42
```

### 3. Train the preference model

```bash
python train_preference.py \
  --model_name /path/to/base_or_sft_model \
  --train_data_path outputs/preference_data \
  --valid_data_path outputs/preference_data \
  --output_dir outputs/preference_run \
  --num_gpus 1
```

### 4. Merge the LoRA adapter

```bash
python merge_lora.py \
  --base /path/to/base_or_sft_model \
  --adapter outputs/preference_run \
  --out outputs/preference_run_merged \
  --dtype bf16
```

## Full Pipeline

For batch execution over the local models included in this repository, use:

```bash
bash pipeline_from_confusion_to_eval_all.sh
```

The SLURM launcher in [apply_server.sh](/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2/apply_server.sh) shows one way to run this pipeline on a multi-GPU cluster environment.

## Notes

- `train_preference.py` uses token-weighted preference training with LoRA.
- `infer_vllm_and_confusion.py` canonicalizes code and sub-code labels before computing confusion edges.
- `pv_utils.py` includes evaluation helpers for code, sub-code, and relaxed span matching.

## Citation

If you use this repository or the released models, please cite the PVminerLLM2 paper:

```bibtex
@article{PVminerLLM2,
  title={PVminerLLM2: Improving Structured Extraction of Patient Voice via Preference Optimization}
}
```
