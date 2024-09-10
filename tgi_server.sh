#!/bin/bash -l
#SBATCH -A p200512
#SBATCH -q default
#SBATCH -p gpu
#SBATCH -t 2:0:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --error="tgi-%j.err"
#SBATCH --output="tgi-%j.out"
#SBATCH --mem=90G


export PMIX_MCA_psec=native
export NUMBA_CACHE_DIR=/tmp
export NUMBA_DEBUG_CACHE=1
export NCCL_DEBUG=INFO

MODEL_REPO="google/gemma-2-2b-it"
HF_TOKEN="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"
APPTAINER="apptainer run --nvccli -B ./data/:/data -B ${PWD} --env HF_TOKEN=$HF_TOKEN "
CONTAINER="text-generation-inference_2.2.0.sif"
TGI="--port 8080 --model-id $MODEL_REPO --num-shard=1 --max-input-length 5000 --max-total-tokens 15000"

echo "HEAD NODE: $(hostname)"
echo "IP ADDRESS: $(hostname --ip-address)"
echo "SSH TUNNEL (HTTP): ssh -p 8822 ${USER}@login.lxp.lu -NL 8002:$(hostname --ip-address):8080"

srun ${APPTAINER} ${CONTAINER} ${TGI}
