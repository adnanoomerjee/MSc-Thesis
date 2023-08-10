#!/bin/bash
#SBATCH --job-name=hct_run
#SBATCH --output=hct_run.out
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:rtx5000:2
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --export=ALL




source ~/.bashrc
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps/high_level_env_gaps/training.py --config 1
