#!/usr/bin/env bash

#SBATCH --job-name=diag-model
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=02:00:00

RUN_ROOTS=${RUN_ROOTS:-"data/train data/val"}
PARAMS_JSON=${PARAMS_JSON:-}
OUTPUT_PATH=${OUTPUT_PATH:-"diagnostics/model_diagnosis.json"}

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "run_roots=$RUN_ROOTS"
echo "params_json=${PARAMS_JSON:-<nominal>}"
echo "output_path=$OUTPUT_PATH"

if [[ -n "$PARAMS_JSON" ]]; then
  uv run -m sysid.diagnose_model $RUN_ROOTS --params-json "$PARAMS_JSON" --output "$OUTPUT_PATH"
else
  uv run -m sysid.diagnose_model $RUN_ROOTS --output "$OUTPUT_PATH"
fi
