source ~/.bashrc
conda activate msc-thesis-hpc

srun --job-name=hct_run \
     --output=hct_run.out \
     --nodes=2 \
     --partition=gpu \
     --ntasks-per-node=1 \
     --time=0-12:00 \
     --mem=100G \
     --gres=gpu:rtx5000:2 \
     --export=ALL \
     python3 /nfs/nhome/live/aoomerjee/MSc-Thesis/training.py --distributed=True