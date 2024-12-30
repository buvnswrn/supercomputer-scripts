#!/bin/bash -l
#SBATCH -A p200512
#SBATCH -q default
#SBATCH -p gpu
#SBATCH -t 2:0:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --error="logs/fine-tune-medical-%j.err"
#SBATCH --output="logs/fine-tune-medical-%j.out"
#SBATCH --mem=90G

export HUGGINGFACE_TOKEN=$(cat ~/hf_auth.env)
export wandb=$(cat ~/wandb_auth.env)

source /project/home/p200512/fine-tuning/bin/activate
python -c  'import sys; print(sys.version)'

python /project/home/p200512/experiments/fine-tuning/llama3-medical.py

#srun ${PYTHON}