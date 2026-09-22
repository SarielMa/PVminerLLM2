#!/bin/bash
#SBATCH --job-name=pv_phase0_qwen35
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_pv_phase0_qwen35_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

# Phase 0 go/no-go for Qwen3.5-9B / Qwen3.8-27B.
# Runs the vllm stage (the hard gate) on a GPU node. The config and modules
# stages need no GPU and can be run on a login node first -- do that before
# burning an allocation, since they are the cheap failures.

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/PV_multiagent/PVminerLLM2"
TEST_PY="${REPO_ROOT}/phase0_load_test.py"
CONDA_ENV="${CONDA_ENV:-finben_qwen35}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"

for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  conda() { return 0; }
  export -f conda
  _FAKE_CONDA_FOR_PURGE=1
fi

module --force purge || true
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

# Keep every large download on scratch, never the project disk.
export HF_HOME="${HF_HOME:-/nfs/roberts/scratch/pi_sjf37/lm2445/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/nfs/roberts/scratch/pi_sjf37/lm2445/.cache/vllm}"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}"

module load miniconda

if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi

conda activate "${CONDA_ENV}"

# The env ships a newer libstdc++ than the nodes (/lib64 tops out at
# CXXABI_1.3.13; the upgraded env pulls in libicui18n.so.78, which needs
# CXXABI_1.3.15). LD_LIBRARY_PATH is set above, before activation, so the
# env lib was not on it -- vLLM's registry subprocess then died on an
# ImportError and surfaced as a bogus ModelConfig ValidationError.
if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

which python
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"
python -c "import transformers, vllm; print('transformers', transformers.__version__, '| vllm', vllm.__version__)"
nvidia-smi

cd "${REPO_ROOT}"
[[ -f "${TEST_PY}" ]] || { echo "Missing ${TEST_PY}" >&2; exit 1; }

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  TP=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  TP=$(python -c "import torch; print(torch.cuda.device_count())")
fi

echo "============================================================"
echo "CONDA_ENV : ${CONDA_ENV}"
echo "MODEL     : ${MODEL}"
echo "TP        : ${TP}"
echo "HF_HOME   : ${HF_HOME}"
echo "============================================================"

python "${TEST_PY}" --model "${MODEL}" --stage all --tp "${TP}"
