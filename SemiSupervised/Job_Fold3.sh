#!/bin/bash
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=400G             
#SBATCH --time=00-02:59:00  # time (DD-HH:MM)
#SBATCH --account=def-guibault
#SBATCH --output=%N-%j.out

echo 'Start'

source ~/ENV/bin/activate
python -u step3_trainingSSKNN.py

echo 'Finish'