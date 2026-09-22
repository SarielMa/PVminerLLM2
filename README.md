# PVminerLLM2

Code for the paper **"PVminerLLM2: Improving Structured Extraction of Patient Voice via Preference Optimization"**.

This repository contains the training and evaluation pipeline used to improve structured extraction of patient voice annotations with preference optimization. The code focuses on three linked stages:

1. Run model inference and build confusion summaries from prediction errors.
2. Convert those confusions into targeted chosen/rejected preference pairs.
3. Train a LoRA-based preference model and optionally merge the adapter for evaluation.

## Released Models

Trained models are available on Hugging Face:

- [PVminerLLM2_1.5B](https://huggingface.co/lm2445/PVminerLLM2_1.5B)
- [PVminerLLM2_3B](https://huggingface.co/lm2445/PVminerLLM2_3B)
- [PVminerLLM2_8B](https://huggingface.co/lm2445/PVminerLLM2_8B)
- [PVminerLLM2_70B](https://huggingface.co/lm2445/PVminerLLM2_70B)

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

## Requirements

### Operating system and hardware

The training and inference pipeline targets **Linux with one or more NVIDIA
GPUs**. vLLM and bitsandbytes do not support macOS or CPU-only execution, so the
pipeline scripts will not run on a Mac. The reported runs used 2 x NVIDIA B200
with CUDA 12.8; smaller models can be run on a single GPU by setting `TP=1`.

`pv_utils.py` and the evaluation metrics are pure Python (`json`, `re`,
`scikit-learn`) and can be imported and run on any platform, which is enough to
re-score existing prediction dumps without a GPU.

### Software environments

Two environments are required, because the Qwen3.5 family needs a newer stack
than the original four models:

| Package | Llama-3.x / Qwen2.5 models | Qwen3.5-9B / Qwen3.8-27B |
| --- | --- | --- |
| Python | 3.11 | 3.11 |
| torch | 2.9.0 | 2.13.0 (cu130) |
| transformers | 4.57.6 | 5.17.0 |
| vLLM | 0.11.2 | 0.30.0 |
| peft | 0.18.1 | 0.18.1 |
| lm-eval | 0.4.11 | 0.4.11 |

The `qwen3_5` architecture is absent from transformers 4.57.6 and vLLM 0.11.2
and present in transformers 5.17.0 and vLLM 0.30.0, so the newer environment is
mandatory for those two models. The older environment is retained for the
original four models so their published numbers stay reproducible; note that
transformers 5 renames `torch_dtype` to `dtype`, which `pv_model_compat.py`
handles for both versions.

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

<!-- ## Citation

If you use this repository or the released models, please cite the PVminerLLM2 paper:

```bibtex
@article{PVminerLLM2,
  title={PVminerLLM2: Improving Structured Extraction of Patient Voice via Preference Optimization}
}
``` -->

## License

The code in this repository is released under the MIT License; see the LICENSE
file for the full text.

The released model weights are covered by the license of their respective base
models:

- PVminerLLM2_1.5B is derived from Qwen2.5-1.5B-Instruct and is released under the
  Apache 2.0 License.
- PVminerLLM2_3B, PVminerLLM2_8B and PVminerLLM2_70B are derived from
  Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct
  respectively, and are subject to the Llama Community License. Built with Llama.

Preference-optimized models derived from Qwen3.5-9B and Qwen3.8-27B are in
preparation. Both base models are released under the Apache 2.0 License, and
the derived weights will be released under the same terms.

The PV-Miner data are not distributed with this repository.
