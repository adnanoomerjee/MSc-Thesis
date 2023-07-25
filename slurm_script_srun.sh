srun --job-name=hct_run --nodes=1 --partition=fast --ntasks-per-node=1 --gres=gpu:1 --exclusive --pty /bin/bash -l

source /nfs/nhome/live/aoomerjee/mambaforge/bin/activate
conda activate msc-thesis-hpc
python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py 
