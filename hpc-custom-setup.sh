cd /nfs/nhome/live/aoomerjee
export HOME=/nfs/nhome/live/aoomerjee
srun -p gpu --gres=gpu:1 -w gpu-380-15 --pty /bin/bash
source .bashrc
conda activate msc-thesis-hpc
