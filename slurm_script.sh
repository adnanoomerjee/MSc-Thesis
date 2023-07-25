#!/bin/bash
#SBATCH --job-name=hct_run
#SBATCH --output=hct_run.out
#SBATCH --nodes=4  
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2


source /nfs/nhome/live/aoomerjee/mambaforge/bin/activate
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py 