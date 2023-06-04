cat << EOOF > script.batch
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -w gpu-380-15
#SBATCH --output=testing.out
#SBATCH --error=testing.err
cd /tmp
export HOME=/tmp
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh -b
./mambaforge/bin/mamba init
source .bashrc
cat <<EOF > environment.yml
name: jax-gpu-env
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.9
  - pip
  - numpy
  - scipy
  - cuda-nvcc<11.6
  - jaxlib=0.4.10=*cuda*
  - jax
EOF
which conda
conda env create -f environment.yml
conda activate jax-gpu-env
python -c "import jax.numpy as jnp; print(jnp.array([1, 2]).device())"
rm -rf /tmp/mambaforge
rm -rf /tmp/.bashrc
rm -rf /tmp/environment.yml
rm -rf /tmp/Mambaforge-$(uname)-$(uname -m).sh
EOOF
sbatch script.batch