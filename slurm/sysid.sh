#!/usr/bin/env bash

#SBATCH --job-name=sysid
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=12:00:00

TRAIN_RUN_ROOT="data/train"
VAL_RUN_ROOT="data/val"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

echo "train_run_root=$TRAIN_RUN_ROOT"
echo "val_run_root=$VAL_RUN_ROOT"

echo "== train analysis =="
uv run -m sysid.analyze_run "$TRAIN_RUN_ROOT"

echo "== train optimize =="
uv run -m sysid.optimize "$TRAIN_RUN_ROOT" --population-size 512 --generations 500 --std-init 2.0

echo "== val analysis =="
uv run -m sysid.analyze_run "$VAL_RUN_ROOT"
