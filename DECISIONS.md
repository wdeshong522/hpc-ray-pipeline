# Decisions and Deviations
 
Rule: `PROPOSAL.md` is the scope. Anything that differs from it, or that the
proposal leaves open, is logged here **before** we build on it.
 
Status values: `open` / `leaning` / `decided` / `deviation (approved)`.
 
Last updated: 2026-11-22 (late night)
 
| # | Topic | Status | Decision / current position |
|---|---|---|---|
| D1 | Problem section says "merged model"; merging 8B models needed 24 GB VRAM in earlier tests (V100s have 16 GB) | open | Serve base + LoRA adapter via vLLM `--enable-lora` instead of merging. Would be a deviation from the proposal's "merged model" wording. Needs approval before Stage 2. |
| D2 | One Docker image, but training and vLLM need different package versions | decided | One image, two virtualenvs: `/opt/venvs/train` and `/opt/venvs/serve`. Neither is on PATH; every script calls its interpreter explicitly. |
| D3 | How a Docker image runs on Wahab | decided | Wahab has Apptainer (no Docker). Build with Docker on the PC, push to a registry, `apptainer pull docker://...` on a compute node, run with `--nv`. Confirmed working. |
| D4 | How responses are sharded across replicas | open | Static contiguous shards (the proposal says "sharded"). A shared queue would balance better but would be a deviation. Output-length variance becomes a measured effect. |
| D5 | Judge calls per response and thinking mode baseline | open | Fix both as config defaults, then sweep them as the proposal's tunables. |
| D6 | Score aggregation function | open | Mean, configurable. |
| D7 | Ray's role beyond LoRA | open | Ray Train for LoRA only. Replicas for generation and judging are SLURM-launched vLLM servers. SLURM accounting and vLLM metrics for those stages. |
| D8 | Public repo and the dataset | decided | Repo is public until the project is turned in, then private. The 4,000-record corpus is synthetic and about 10 MB, so it is committed under `data/` (the `data/*` rule is removed from `.gitignore`). Tokens, model files, results and SLURM logs stay ignored. |
| D9 | Wahab size: proposal says 18 GPU nodes / 72 V100; `sinfo` shows 16 nodes in `gpu` and `timed-gpu` | decided | 16 nodes, 64 V100s. Fix the numbers in the presentation. Scaling plan needs at most 24 GPUs (6 nodes), so it is unaffected. At the time of checking, 9 of 16 nodes were in use by other jobs. |
| D10 | Partition choice: `timed-gpu` has a 2 h limit, `gpu` has 14 days | leaning | Use `gpu` for any run that may exceed 2 h (the 1-GPU LoRA baseline is estimated at about 2.5 to 3 h). Use `main` (CPU) for non-GPU work such as image conversion. |
| D11 | Record SLURM-granted GPUs and nodes in timing records | decided | No. Configuration under test lives in `params`; only job ID, name, nodelist and CPUs per task are captured. |
| D12 | Student model variant | decided | `meta-llama/Llama-3.1-8B-Instruct`. Fits the proposal's "Llama 3.1 8B" and avoids base-model chat-token and stop-token risks. Hugging Face access confirmed. |
| D13 | Judge model variants | decided | `Qwen/Qwen3-8B` and `ibm-granite/granite-4.1-8b`, both chat-tuned, used unmodified. The Granite page lists 8B on the card but 9B in metadata, so verify the real parameter count when sizing 8-bit. |
| D14 | Code in the image | decided | Bind-mounted during development. A release image with the code baked in is built once at the end. |
| D15 | Base image | decided | `python:3.12-slim-bookworm`. CUDA libraries come from pip wheels and the driver from `--nv`. Includes a C compiler (Triton / `torch.compile`) and InfiniBand user-space libraries (for NCCL). |
| D16 | Registry | decided | Docker Hub (`wdeshong/hpc-ray-pipeline`), public. Already proven to work from Wahab. |
| D17 | Serve environment: numpy and CUDA pins | decided | Dropped from `constraints-serve.txt`. The validated env's numpy 2.5.2 contradicts numba and mistral-common metadata, and its CUDA pins belong to the post-swap torch. The image carries a numpy compatible with numba; verified working on a V100. |
| D18 | Image `wdeshong/hpc-ray-pipeline:0.1` | decided | Built and validated: `hpc.sif` is 8.8 GB; both venvs pass `verify_env.py --gpu` on a Wahab V100 (compute capability 7.0, `sm_70` present). |
| D19 | Judging case: 8-bit (1 GPU/replica) vs pipeline-parallel (2 GPUs/replica) | decided | Pipeline-parallel only, as the proposal specifies when 8-bit is not validated (not a deviation). Tested on a V100 with vLLM 0.30.0: fp16 Qwen3-8B loads 15.27 GiB and runs out of memory on 15.77 GiB; `bitsandbytes` is not a vLLM method; every on-the-fly 8-bit method (`fp8*`, `int8_per_channel_weight_only`, `torchao`, `online`, `awq`) needs compute capability 75+ (V100 is 70). Only GPTQ/compressed-tensors pass, and they need pre-quantized checkpoints (none known for Granite 4.1 8B), which would be new scope. Judge units are 6 GPUs (2 Qwen replicas + 1 Granite replica, 2 GPUs each): 6, 12, 18, 24 GPUs. The 24-GPU point needs 6 idle nodes, so schedule it for a quiet window. Scripts: `experiments/smoke_8bit.py`, `experiments/quant_capabilities.py`. |
| D20 | LoRA base precision | decided | LoRA on a 4-bit NF4 base (QLoRA), as in the existing code. 8B fp16 weights (~15-16 GB) cannot fit a 16 GB V100 for training. bitsandbytes 4-bit already validated in the train venv. State on the methods slide. |
| D21 | 1-GPU LoRA case path | decided | Run the 1-GPU case through Ray Train with one worker, not the existing no-Ray shortcut, so T(1) includes the same Ray overhead as T(2,4,8) and speedup is a fair comparison. |
| D22 | Ray worker placement | decided | `PACK`, not `STRICT_PACK`. 8 GPUs spans two 4-GPU nodes; `STRICT_PACK` cannot place that. |
| D23 | Validation split and eval during LoRA training | decided | No train/eval split and no mid-run eval or mid-run checkpoints. Train on all N examples (4,000 at the top strong-scaling point), matching the proposal's workloads exactly. Sanity check is the logged training loss. No quality claim across GPU counts (global batch grows with GPUs; learning rate not retuned). |
| D24 | Startup vs processing boundary (all training runs) | decided | Startup = Ray cluster start, worker launch, tokenizer, dataset prep, model load, LoRA setup (everything before the first training step). Processing = `trainer.train()` only. Same rule as the vLLM stages, where model load is startup. Model load is replicated per worker and does not scale, so it must not sit inside processing. |
| D25 | Five malformed lines in the training corpus | decided (pending Will's confirmation) | Leave data files untouched. Malformed or unusable lines are skipped and reported (not hidden); each run records the skipped count. Pool is exactly 4,000 unique records (verified 2026-11-22: good=4000, unique=4000, 5 bad lines in domain10_batch11, domain10_batch15 x2, domain1_batch14, domain6_batch11), matching the proposal's top workload. Repairing the lines would grow the pool to 4,005 and change every prefix, so it is out of scope. |
 
## Open items that are not yet decisions
 
- **Queue timing for the 24-GPU judging point:** needs 6 idle GPU nodes; pick a quiet window and check `sinfo` beforehand.
- **Startup measurement rule:** first Qwen3-8B load took 366 s (215 s was a first-time download, 142 s NFS read). Decide whether to report cold or warm cache, and pre-stage weights before timed runs.
- **Strong-scaling workload sizes:** fixed by pilot runs (1-GPU baselines must finish in practical time).
- **Training cost:** extrapolated from earlier Granite timings (not measured for Llama); confirm in the Phase 2 pilot.
- **Ray cluster startup:** one Ray head on the first node and one worker per other node, started inside the SLURM job via `srun` + `apptainer exec --nv`, before the driver connects. Per-job unique ports.
- **Ray dashboard:** the existing code sets `include_dashboard=False` (log spam on Turing). The proposal's analysis uses the Ray dashboard/timeline. Decide how to get that data without the noise.
## Cluster notes (working knowledge, not decisions)
 
- **Two-node NCCL (checked 2026-11-22):** inside the container, NCCL 2.20.5 uses InfiniBand (`NET/IB`, mlx5_0) between nodes; no TCP fallback. Measured 3.93 GB/s bus bandwidth for a 256 MB fp32 all-reduce on 2 nodes x 1 GPU (`experiments/nccl_check.py`). GPU Direct RDMA is disabled on Wahab, so GPU traffic is staged through host memory. The `ibv_set_ece` warnings are harmless.
- Wahab's login shell is csh/tcsh. Run `bash` first for `export` and `VAR=value` syntax.
- Heavy jobs (for example converting a container image) must run as batch jobs on compute nodes, not on a login node.
- `!` typed at an interactive prompt can trigger history expansion; prefer `sbatch --wrap` or the file editor over heredocs containing `#!`.
 