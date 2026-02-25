#!/usr/bin/env bash

#SBATCH --job-name=environment_smoke
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=30:00

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl

uv run -m environment.environment
