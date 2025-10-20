#!/bin/bash

# Usage: sbatch scripts/slurm.sh [CONFIG_PATH] [ADDITIONAL_ARGS...]

#SBATCH --job-name=lacuna
#SBATCH --partition=a3ultra
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

START_TIME=$(date +%s)
echo "JOB STARTED, TIME: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
echo "JOB NAME: $SBATCH_JOB_NAME"
echo "NODE LIST: $SLURM_JOB_NODELIST"

set -eo pipefail
set -x

source .env

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export UV_CACHE_DIR="$HOME/.cache/uv"

export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0  # enable for debugging
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ $NNODES -eq 1 ]; then
    # Single-node: use uv run train directly
    export CMD="uv run train $@"
else
    NNODES=$SLURM_NNODES
    NODE_RANK=$SLURM_NODEID
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    MASTER_PORT=29500

    echo "NNODES: $NNODES"
    echo "NODE_RANK: $NODE_RANK"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"

    # Multi-node: use uv run train with torchrun args
    export CMD="uv run train $@ \
        --trainer.run_name $SBATCH_JOB_NAME \
        --torchrun.nnodes $NNODES \
        --torchrun.node_rank $NODE_RANK \
        --torchrun.master_addr $MASTER_ADDR \
        --torchrun.master_port $MASTER_PORT"
fi

echo "RUNNING CMD: ${CMD}"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

# bash -c needed for delayed interpolation of env vars
srun $SRUN_ARGS bash -c "$CMD"

echo "JOB FINISHED, TIME: $(date)"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINUTES=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))
echo "TOTAL ELAPSED TIME: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s"
