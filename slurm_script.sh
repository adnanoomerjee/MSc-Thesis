#!/bin/bash
#SBATCH --job-name=hct_run
#SBATCH --output=hct_run.out
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-10:00
#SBATCH --mem=10G      
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --exclude=gpu-sr670-20,gpu-sr670-22


source ~/.bashrc
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps/flat_env_mlp/training.py --config $1
