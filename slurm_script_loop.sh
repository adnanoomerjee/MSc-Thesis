#!/bin/bash
#SBATCH --job-name=hct_run0
#SBATCH --output=hct_run0.out
#SBATCH --nodes=1 
#SBATCH --partition=medium
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2


source /nfs/nhome/live/aoomerjee/mambaforge/bin/activate
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py --config=0
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py --config=1