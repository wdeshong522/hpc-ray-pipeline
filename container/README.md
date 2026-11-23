# Container

One image, two virtualenvs (see `Dockerfile` header). Status: **untested until the first build**.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Build recipe |
| `requirements-train.txt` / `requirements-serve.txt` | Direct dependencies, versions from the validated Wahab envs |
| `constraints-train.txt` / `constraints-serve.txt` | Every `==` pin from those envs, applied with `pip -c` so transitive packages keep validated versions |

## Regenerating the constraints files

From `pip freeze` output of the validated envs (conda-built `@ file://` lines are dropped; those packages are resolved by pip normally). The serve constraints also drop numpy and all torch/CUDA-owned packages, because the validated env's numpy 2.5.2 contradicts numba/mistral-common metadata and its CUDA pins belong to the post-swap torch (see DECISIONS.md D17):

```
grep -v ' @ ' freeze_train.txt | grep '==' | sort > container/constraints-train.txt
grep -v ' @ ' freeze_vllm.txt  | grep '==' | grep -v -E '^(torch|torchvision|torchaudio|numpy|cuda-bindings|cuda-python|cuda-core|cuda-pathfinder|cuda-toolkit|cuda-tile|triton|nccl4py)==|^nvidia-' | sort > container/constraints-serve.txt
```

## Build and push (on the PC)

```
docker build -f container/Dockerfile -t <registry>/hpc-ray-pipeline:0.1 .
docker push <registry>/hpc-ray-pipeline:0.1
```

## Use on Wahab

```
apptainer pull hpc.sif docker://<registry>/hpc-ray-pipeline:0.1
A="-A odu -p timed-gpu --gres=gpu:1 -c 4 -t 10"
srun $A apptainer exec --nv hpc.sif /opt/venvs/train/bin/python /opt/verify_env.py train --gpu
srun $A apptainer exec --nv hpc.sif /opt/venvs/serve/bin/python /opt/verify_env.py serve --gpu
```
