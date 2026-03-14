#!/usr/bin/env bash

#SBATCH --job-name=train
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=1-00:00:00

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

uv run -m rl.train --num-devices 4
