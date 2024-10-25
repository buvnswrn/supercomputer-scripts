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
NODE_IP=$(hostname --ip-address)


#MODEL_REPO="Alibaba-NLP/gte-Qwen2-7B-instruct"
MODEL_REPO="jinaai/jina-embeddings-v2-base-de"
#MODEL_REPO="Alibaba-NLP/gte-multilingual-base"
#revision="refs/pr/7"
#MODEL_REPO="sentence-transformers/LaBSE"
#MODEL_REPO="Salesforce/SFR-Embedding-Mistral"
#MODEL_REPO="intfloat/multilingual-e5-large-instruct"
#MODEL_REPO="jinaai/jina-embeddings-v3"

TGI_SERVER_IP=$(cat ~/tgi_server_ip.env)
echo "TGI Server is at: $TGI_SERVER_IP"
CLUSTER_REGISTER_ENDPOINT="http://$TGI_SERVER_IP:8000/other_services/api/v1/register_cluster"
echo "Registering Server at $CLUSTER_REGISTER_ENDPOINT"

HF_TOKEN=$(cat ~/hf_auth.env)
APPTAINER="apptainer run --nvccli -B ./data/:/data --env HF_TOKEN=$HF_TOKEN "
CONTAINER="text-embeddings-inference_1.5.sif"
TEI="--port 8080 --model-id $MODEL_REPO"
#TEI="--port 8080 --model-id $MODEL_REPO --revision=$revision"

#echo "Using dynamic port: $PORT"
echo "HEAD NODE: $(hostname)"
echo "IP ADDRESS: $(hostname --ip-address)"
#echo "Using port: $NODE_PORT on node $SLURM_NODEID"
echo "SSH TUNNEL (HTTP): ssh -p 8822 ${USER}@login.lxp.lu -NL 8002:$(hostname --ip-address):8080"
curl -X POST "$CLUSTER_REGISTER_ENDPOINT?ip=$NODE_IP"

srun ${APPTAINER} ${CONTAINER} ${TEI}
