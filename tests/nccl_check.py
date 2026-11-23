"""Two-node NCCL check: which transport carries GPU traffic between nodes?

Proposal link: LoRA data-parallel scaling past 4 GPUs crosses nodes, and its
gradient all-reduce runs over NCCL.  This checks the same path inside the
container, with one GPU per node.  Run under SLURM (see slurm/nccl_check.sbatch).
Read the transport from the NCCL_DEBUG lines ("NET/IB" vs "NET/Socket").
"""
import os
import time

import torch
import torch.distributed as dist

rank = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])

torch.cuda.set_device(0)
# MASTER_ADDR / MASTER_PORT come from the sbatch file (env:// is the default)
dist.init_process_group("nccl", rank=rank, world_size=world)

n_floats = 64 * 1024 * 1024                    # 256 MB of float32
x = torch.ones(n_floats, device="cuda")
size_bytes = n_floats * 4

for _ in range(3):                             # warm-up (connection setup)
    dist.all_reduce(x)
torch.cuda.synchronize()

iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    dist.all_reduce(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / iters

# ring all-reduce moves 2*(n-1)/n of the buffer per rank
busbw_gb = 2 * (world - 1) / world * size_bytes / dt / 1e9
if rank == 0:
    print(f"NCCLCHECK world={world} per_op_s={dt:.4f} bus_bw_GBps={busbw_gb:.2f}")
dist.destroy_process_group()
