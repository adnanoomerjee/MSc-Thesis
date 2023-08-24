#!/bin/bash
#SBATCH --job-name=hct_run
#SBATCH --output=hct_run.out
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --exclude=gpu-380-[11-14],gpu-sr670-[20-22]
#SBATCH --export=ALL

source ~/.bashrc
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/testing.py