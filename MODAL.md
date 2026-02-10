# Running modded-nanogpt on Modal

## Cost

H100 on Modal: ~$3.95/hr (per-second billing). A single training run (~6 min on 8xH100) will take longer on 1xH100 due to gradient accumulation, but you only pay for what you use.

## One-time setup

Install the Modal CLI and authenticate:

```bash
pip install modal
modal setup
```

## Dev loop

### 1. Start an interactive GPU shell

```bash
modal shell modal_config.py
```

This gives you a bash shell on an H100 with:
- All pip dependencies pre-installed (cached in the image layer — no reinstall)
- Triton/inductor/nvcc kernel caches on persistent volumes (survives across sessions)
- A data volume at `/mnt/data` for the training dataset

### 2. Clone the repo

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
```

### 3. Download training data (first time only)

The data volume persists across sessions. First run, download into it:

```bash
# Create the expected directory structure on the persistent volume
mkdir -p /mnt/data/data/fineweb10B

# Download 900M tokens (~9 shards). The script skips files that already exist,
# so re-running this is safe and will only download missing shards.
DATA_PATH=/mnt/data python data/cached_fineweb10B.py 9
```

On subsequent sessions, the data is already there — skip this step.

### 4. Run training

```bash
# 1 GPU — the code handles world_size=1 via gradient accumulation (8 accum steps)
DATA_PATH=/mnt/data torchrun --standalone --nproc_per_node=1 train_gpt.py
```

The first run will be slower due to Triton/inductor kernel compilation. These compiled kernels are saved to the persistent cache volumes, so subsequent runs start much faster.

## Quick reference

```bash
# Full dev loop after first-time data download:
modal shell modal_config.py
# then inside the shell:
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
DATA_PATH=/mnt/data torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## What's cached where

| What | Where | Persists? |
|---|---|---|
| pip packages | Baked into Modal image | Yes (rebuilt only when `modal_config.py` changes) |
| Training data (fineweb) | `/mnt/data` volume | Yes |
| Triton compiled kernels | `/root/.triton` volume | Yes |
| torch.compile / inductor cache | `/root/.inductor-cache` volume | Yes |
| CUDA compiler cache | `/root/.nv` volume | Yes |

## Tips

- **Minimize idle time**: Modal bills per-second. Start the shell, run your experiment, exit. Don't leave shells open.
- **Data download is the bottleneck on first run**: ~900M tokens takes a few minutes from HuggingFace. After that it's cached on the volume.
- **Kernel compilation is the bottleneck on second run**: First training launch compiles Triton/inductor kernels. After that the cache volumes make subsequent launches fast.
- **Scaling up**: To use multiple GPUs, change `gpu="H100"` to `gpu="H100:8"` in `modal_config.py` and use `--nproc_per_node=8` in the torchrun command.
- **Check volume contents**: `modal volume ls nanogpt-data` to see what's stored.
- **Nuke caches if something breaks**: `modal volume rm triton-cache /` etc. to clear a cache volume.
