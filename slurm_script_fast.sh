#!/bin/bash
#SBATCH --job-name=hct_run_fast
#SBATCH --output=hct_run_fast.out
#SBATCH --nodes=1 
#SBATCH --partition=fast
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive

source /nfs/nhome/live/aoomerjee/mambaforge/bin/activate
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/debug.py