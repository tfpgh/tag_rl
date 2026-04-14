#!/usr/bin/env bash

#SBATCH --job-name=sysid
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --time=1-00:00:00

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

uv run -m sysid.fit --data-root data --output-dir runs/sysid_$SLURM_JOB_ID --generations 10000 --population-size 256
