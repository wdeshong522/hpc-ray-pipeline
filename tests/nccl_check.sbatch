#!/bin/bash
#SBATCH -A odu
#SBATCH -p timed-gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 15
#SBATCH -o /home/%u/nccl_%j.log

# Rank 0 lives on the first node of the allocation; port is unique per job.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# Make NCCL report which transport it picked.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# One task per node; each runs the training venv inside the container.
srun apptainer exec --nv ~/sif/hpc.sif \
    /opt/venvs/train/bin/python ~/nccl_check.py
