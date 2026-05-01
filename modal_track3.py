import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "modded-nanogpt-track3"
REMOTE_REPO_DIR = Path("/root/modded-nanogpt")

# Set these before `modal run` if you want a different machine shape:
#   MODAL_GPU_TYPE=A100 MODAL_NUM_GPUS=2 modal run modal_track3.py
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "H100")
NUM_GPUS = int(os.environ.get("MODAL_NUM_GPUS", "1"))
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", str(2 * 60 * 60)))

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.10",
        "numpy",
        "tqdm",
        "huggingface-hub",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
    )
    .env(
        {
            "TRITON_CACHE_DIR": "/root/.triton",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # copy=False keeps the dev loop fast: Modal ships the current local files at
    # container startup instead of rebuilding the image for every optimizer edit.
    .add_local_dir(
        ".",
        remote_path=str(REMOTE_REPO_DIR),
        copy=False,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".DS_Store",
            "data/fineweb10B",
            "fineweb10B",
            "logs",
            "checkpoints",
        ],
    )
)

volumes = {
    str(REMOTE_REPO_DIR / "data" / "fineweb10B"): modal.Volume.from_name(
        "nanogpt-fineweb10b", create_if_missing=True
    ),
    str(REMOTE_REPO_DIR / "logs"): modal.Volume.from_name(
        "nanogpt-track3-logs", create_if_missing=True
    ),
    str(REMOTE_REPO_DIR / "checkpoints"): modal.Volume.from_name(
        "nanogpt-track3-checkpoints", create_if_missing=True
    ),
    "/root/.triton": modal.Volume.from_name(
        "nanogpt-track3-triton-cache", create_if_missing=True
    ),
    "/root/.nv": modal.Volume.from_name(
        "nanogpt-track3-nv-cache", create_if_missing=True
    ),
    "/root/.inductor-cache": modal.Volume.from_name(
        "nanogpt-track3-inductor-cache", create_if_missing=True
    ),
}


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    timeout=TIMEOUT_SECONDS,
    volumes=volumes,
)
def train(
    script: str = "records/track_3_optimization/train_gpt_simple.py",
    num_data_shards: int = 40,
    train_steps: int = 0,
    val_interval: int = 0,
    requested_gpu_type: str = GPU_TYPE,
    requested_num_gpus: int = NUM_GPUS,
    extra_args: str = "",
):
    """Run the Track 3 benchmark on Modal."""
    os.chdir(REMOTE_REPO_DIR)
    import torch

    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError("No CUDA GPUs are visible inside the Modal container")
    print(f"Modal GPU request: {requested_gpu_type}:{requested_num_gpus}")
    print(f"Visible CUDA GPUs: {visible_gpus}")
    try:
        subprocess.run(["nvidia-smi", "-L"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found")
    if visible_gpus != requested_num_gpus:
        print(
            f"WARNING: requested {requested_num_gpus} GPU(s), but container sees {visible_gpus}"
        )

    script_path = REMOTE_REPO_DIR / script
    if not script_path.exists():
        raise FileNotFoundError(f"No training script at {script_path}")
    if train_steps:
        if train_steps < 1:
            raise ValueError("train_steps must be 0 or a positive integer")
        print(f"Overriding train_steps to {train_steps}")
    if val_interval:
        if val_interval < 1:
            raise ValueError("val_interval must be 0 or a positive integer")
        print(f"Overriding val_interval to {val_interval}")

    try:
        print(f"Using local source uploaded to {REMOTE_REPO_DIR}")
        print(f"Preparing FineWeb10B shard cache with {num_data_shards} train shards")
        subprocess.run(
            ["python", "data/cached_fineweb10B.py", str(num_data_shards)],
            check=True,
        )

        command = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={visible_gpus}",
            script,
            *shlex.split(extra_args),
        ]
        print("+ " + " ".join(shlex.quote(part) for part in command))
        env = os.environ.copy()
        if train_steps:
            env["NANOGPT_TRAIN_STEPS"] = str(train_steps)
        if val_interval:
            env["NANOGPT_VAL_INTERVAL"] = str(val_interval)
        env["NANOGPT_CHECKPOINT_DIR"] = str(REMOTE_REPO_DIR / "checkpoints")
        subprocess.run(command, env=env, check=True)
    finally:
        logs = sorted(
            (REMOTE_REPO_DIR / "logs").glob("*.txt"), key=lambda p: p.stat().st_mtime
        )
        if logs:
            print(f"Latest log: {logs[-1]}")

        for volume in volumes.values():
            volume.commit()


@app.local_entrypoint()
def main(
    script: str = "records/track_3_optimization/train_gpt_simple.py",
    num_data_shards: int = 40,
    train_steps: int = 0,
    val_interval: int = 0,
    extra_args: str = "",
):
    call = train.spawn(
        script=script,
        num_data_shards=num_data_shards,
        train_steps=train_steps,
        val_interval=val_interval,
        requested_gpu_type=GPU_TYPE,
        requested_num_gpus=NUM_GPUS,
        extra_args=extra_args,
    )
    print(f"Spawned train call: {call.object_id}")
