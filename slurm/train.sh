#!/usr/bin/env bash

#SBATCH --job-name=train
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1-00:00:00

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl

uv run -m rl.train
