#!/usr/bin/env bash

#SBATCH --job-name=tag_train
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gpus=8
#SBATCH --constraint=GPU_SKU:RTX6000


export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

# module load cuda/13.0.2
# module load cudnn/9.17.1-cuda13

uv sync --extra cuda13

uv run -m rl.train --num-devices 8
