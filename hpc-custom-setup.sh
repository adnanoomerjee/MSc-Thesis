cd /nfs/nhome/live/aoomerjee
export HOME=/nfs/nhome/live/aoomerjee
srun -p gpu --gres=gpu:2 -w gpu-380-15 --exclusive --pty /bin/bash -l
srun -p gpu --gres=gpu:4 -w gpu-sr670-21 --exclusive --pty /bin/bash -l

source .bashrc
conda activate msc-thesis-hpc
