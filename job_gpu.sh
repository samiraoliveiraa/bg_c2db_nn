#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/.bashrc
module load cuda/12.2
conda activate ilumpy

python -u treinamento_da_rede.py > treinamento_da_rede_versao01_gpu.out

