#!/usr/bin/env bash

#SBATCH --job-name=sysid_compare
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --constraint=GPU_ANY

if [ "$#" -eq 0 ]; then
    printf 'Usage: sbatch slurm/sysid_render_compare_vacc.sh --run-dir <path> --params <path> --output <path> [extra args...]\n' >&2
    exit 1
fi

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true

uv sync --extra cuda13

uv run -m sysid.render_compare --run-dir data/eval --params runs/sys_id_something/history.jsonl --output render_compare.mp4
