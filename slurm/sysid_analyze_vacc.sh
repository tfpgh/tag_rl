#!/usr/bin/env bash

#SBATCH --job-name=sysid_analyze
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --constraint=GPU_ANY

if [ "$#" -eq 0 ]; then
    printf 'Usage: sbatch slurm/sysid_analyze_vacc.sh --params <path> --output-dir <path> [extra args...]\n' >&2
    exit 1
fi

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

uv sync --extra cuda13

uv run -m sysid.analyze_physics "$@"
