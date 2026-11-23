# Hybrid-parallel dental hygiene pipeline: one image, two virtualenvs.
#   /opt/venvs/train : Stage 1 LoRA fine-tuning (Ray Train, PEFT, TRL)
#   /opt/venvs/serve : Stages 2-3 serving and judging (vLLM)
# Neither venv is on PATH: every script calls its interpreter explicitly,
# e.g. /opt/venvs/train/bin/python, so there is never doubt about which
# environment is running.
#
# Build from the repo root:
#   docker build -f container/Dockerfile -t <registry>/hpc-ray-pipeline:0.1 .

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# --- System packages ------------------------------------------------------
# build-essential : Triton and torch.compile compile small C stubs at run time
# libgomp1        : OpenMP runtime used by PyTorch CPU code
# libnuma1        : NUMA library used by several CUDA/NCCL components
# libibverbs1, ibverbs-providers, librdmacm1 : InfiniBand user-space libraries,
#                   so NCCL can use the cluster's InfiniBand network instead of
#                   falling back to TCP sockets (proposal: multi-node runs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl \
        libgomp1 libnuma1 \
        libibverbs1 ibverbs-providers librdmacm1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venvs/train && python -m venv /opt/venvs/serve

# --- Training environment (Stage 1) ---------------------------------------
COPY container/requirements-train.txt container/constraints-train.txt /tmp/build/
RUN /opt/venvs/train/bin/pip install \
        -r /tmp/build/requirements-train.txt \
        -c /tmp/build/constraints-train.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124

# --- Serving environment (Stages 2-3) -------------------------------------
# Step 1: install vLLM. This pulls its own default torch (a CUDA 13 build).
# constraints-serve.txt deliberately EXCLUDES numpy and every torch/CUDA-owned
# package (nvidia-*, cuda-*, triton, nccl4py): those pins come from the final
# env AFTER step 2 below, and conflict with the default torch pulled here.
COPY container/requirements-serve.txt container/constraints-serve.txt /tmp/build/
RUN /opt/venvs/serve/bin/pip install \
        -r /tmp/build/requirements-serve.txt \
        -c /tmp/build/constraints-serve.txt

# Step 2: replace torch with the cu126 builds so V100 (sm_70) kernels exist.
# All three must be replaced together, or torchvision's CUDA check fails.
RUN /opt/venvs/serve/bin/pip install --force-reinstall \
        "torch==2.13.0+cu126" "torchvision==0.28.0+cu126" "torchaudio==2.11.0+cu126" \
        --extra-index-url https://download.pytorch.org/whl/cu126

# Informational only (does not fail the build): lists any dependency
# metadata that the torch swap left inconsistent.
RUN /opt/venvs/serve/bin/pip check || true

# --- Build-time checks (no GPU on the build machine) ------------------------
# Fail the build if package versions drifted or the torch binaries were not
# compiled for V100 (sm_70). The real GPU test runs later, on Wahab:
#   apptainer exec --nv hpc.sif /opt/venvs/serve/bin/python /opt/verify_env.py serve --gpu
COPY container/verify_env.py /opt/verify_env.py
RUN /opt/venvs/train/bin/python /opt/verify_env.py train
RUN /opt/venvs/serve/bin/python /opt/verify_env.py serve

RUN rm -rf /tmp/build

LABEL org.opencontainers.image.title="hpc-ray-pipeline" \
      org.opencontainers.image.description="LoRA (Ray Train) + vLLM generation + two-judge panel, V100/Wahab"

# Project code is NOT baked in during development: it is bind-mounted at run
# time (apptainer exec --bind <repo>:/workspace ...). A release image that
# copies the code in is built once at the end (see DECISIONS.md D14).
WORKDIR /workspace
