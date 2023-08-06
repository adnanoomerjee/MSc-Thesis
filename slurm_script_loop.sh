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

'''srun --job-name=hct_run -w gpu-erlich01 --nodes=1 --partition=fast --ntasks-per-node=1 --gres=gpu:2 --exclusive --pty /bin/bash -l
#sbatch --job-name=hct_run -w gpu-erlich01 --nodes=1 --partition=fast --ntasks-per-node=1 --gres=gpu:2 --exclusive --pty /bin/bash -l

#srun --job-name=hct_run -w gpu-erlich01 --nodes=1 --ntasks-per-node=1 --gres=gpu:2 --exclusive --pty /bin/bash -l

source /nfs/nhome/live/aoomerjee/mambaforge/bin/activate
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py 
srun --job-name=hct_run -w gpu-erlich01 --nodes=1 --partition=fast --ntasks-per-node=1 --gres=gpu:2 --exclusive --pty /bin/bash -l
srun --job-name=hct_run --nodes=1 --partition=gpu --ntasks-per-node=1 --gres=gpu:4 --exclusive --pty /bin/bash -l
srun --job-name=hct_run --nodes=4 --partition=gpu --ntasks-per-node=1 --gres=gpu:2 --pty /bin/bash -l
'''
