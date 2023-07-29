#!/bin/bash
#SBATCH --job-name=hct_run
#SBATCH --output=hct_run.out
#SBATCH --nodes=1 
#SBATCH --partition=fast
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpu-erlich01
#SBATCH --exclusive

source /nfs/nhome/live/aoomerjee/mambaforge/bin/activate
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py --config=0
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py --config=1