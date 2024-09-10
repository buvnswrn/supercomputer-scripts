#!/bin/bash -l
#SBATCH -A p200512
#SBATCH -q default
#SBATCH -p gpu
#SBATCH -t 2:0:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --error="logs/tei-%j.err"
#SBATCH --output="logs/tei-%j.out"
#SBATCH --mem=90G

export PMIX_MCA_psec=native
export NUMBA_CACHE_DIR=/tmp
export NUMBA_DEBUG_CACHE=1
export NCCL_DEBUG=INFO
#PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]);s.close()')
#NODE_IP=$(hostname --ip-address)


MODEL_REPO="BAAI/bge-large-en-v1.5"
HF_TOKEN=$(cat ~/hf_auth.env)
APPTAINER="apptainer run --nvccli -B ./data/:/data --env HF_TOKEN=$HF_TOKEN "
CONTAINER="text-embeddings-inference_1.5.sif"
TEI="--port 8080 --model-id $MODEL_REPO"

#echo "Using dynamic port: $PORT"
echo "HEAD NODE: $(hostname)"
echo "IP ADDRESS: $(hostname --ip-address)"
#echo "Using port: $NODE_PORT on node $SLURM_NODEID"
echo "SSH TUNNEL (HTTP): ssh -p 8822 ${USER}@login.lxp.lu -NL 8002:$(hostname --ip-address):8080"

srun ${APPTAINER} ${CONTAINER} ${TEI}
