#!/bin/bash
#SBATCH --job-name=hct_run
#SBATCH --output=hct_run.out
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-6:00
#SBATCH --mem=10G      
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --export=ALL


source ~/.bashrc
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/testing.py #--config $1
