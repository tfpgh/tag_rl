#!/usr/bin/env bash

#SBATCH --job-name=sysid_deep
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:RTX6000

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

uv sync --extra cuda13

uv run -m sysid.deep_analysis \
  --data-root data \
  --params runs/sysid_4015155/history.jsonl \
  --history-source global_best \
  --output-dir runs/sysid_deep_analysis_$SLURM_JOB_ID \
  --window-seconds 1.5 \
  --preroll-seconds 0.5 \
  --stride-seconds 0.5 \
  --sensitivity-points 7 \
  --local-fraction 0.30 \
  --top-k-windows 30 \
  --top-k-runs 15 \
  --save-sample-details
