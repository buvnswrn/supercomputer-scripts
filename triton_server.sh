#!/bin/bash -l
#SBATCH -A p200512
#SBATCH -q dev
#SBATCH -p gpu
#SBATCH --res gpudev
#SBATCH -t 2:0:0
#SBATCH -N 1
#SBATCH --gpus-per-task=4
#SBATCH --error="triton-%j.err"
#SBATCH --output="triton-%j.out"

export PMIX_MCA_psec=native

MODEL_REPO="mistralai/Mistral-7B-v0.1"
APPTAINER="apptainer run --nvccli -B ${PWD} "
CONTAINER="tritonserver_24.05-trtllm-python-py3.sif"
TRITON="tritonserver --model-repository=tensorrtllm_backend/all_models/inflight_batcher_llm --exit-on-error=false --strict-readiness=false"

echo "HEAD NODE: $(hostname)"
echo "IP ADDRESS: $(hostname --ip-address)"
echo "SSH TUNNEL (HTTP): ssh -p 8822 ${USER}@login.lxp.lu  -NL 8002:$(hostname --ip-address):8000" 
echo "SSH TUNNEL (GRPC): ssh -p 8822 ${USER}@login.lxp.lu  -NL 8003:$(hostname --ip-address):8001" 

srun ${APPTAINER} ${CONTAINER} ${TRITON}
