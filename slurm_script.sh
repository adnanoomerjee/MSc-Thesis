#!/bin/bash
#SBATCH --job-name=hct_run
#SBATCH --output=hct_run.out
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --exclude=gpu-sr670-20,gpu-380-[11-14]

source ~/.bashrc
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps_v3/gaps/hma3/run.py --config $1

