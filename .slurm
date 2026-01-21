#!/bin/bash
#SBATCH --job-name=dqn-training
#SBATCH --output=training_output.txt
#SBATCH --gres=gpu:1  # Usar 1 GPU
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

module load cuda/11.2
conda activate nuevo_entorno_py310
python train_deep_rl_dqn.py
