#!/usr/bin/env bash

#SBATCH --job-name=train
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1-00:00:00

export TQDM_MININTERVAL=15
export PYTHONUNBUFFERED=1
uv run -m rl.train
