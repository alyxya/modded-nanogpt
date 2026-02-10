import os
import modal

NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))

app = modal.App("modded-nanogpt")

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
    .env({
        "TRITON_CACHE_DIR": "/root/.triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    })
)

volumes = {
    "/mnt/data": modal.Volume.from_name("nanogpt-data", create_if_missing=True),
    "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
    "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
    "/root/.inductor-cache": modal.Volume.from_name("inductor-cache", create_if_missing=True),
}

REPO = "https://github.com/alyxya/modded-nanogpt.git"
REPO_DIR = "/root/modded-nanogpt"

@app.function(image=image, gpu=f"H100:{NUM_GPUS}", timeout=3600, volumes=volumes)
def train(num_data_shards: int = 9):
    """One-command training run: modal run modal_config.py::train"""
    import subprocess, os, shutil

    # Clone repo
    subprocess.run(["git", "clone", REPO, REPO_DIR], check=True)

    # Symlink data dir to persistent volume so downloads are cached
    os.makedirs("/mnt/data/data/fineweb10B", exist_ok=True)
    os.symlink("/mnt/data/data/fineweb10B", f"{REPO_DIR}/data/fineweb10B")

    # Download data (idempotent — skips files that already exist on the volume)
    subprocess.run(
        ["python", "data/cached_fineweb10B.py", str(num_data_shards)],
        cwd=REPO_DIR, check=True,
    )

    # Train
    subprocess.run(
        ["torchrun", "--standalone", f"--nproc_per_node={NUM_GPUS}", "train_gpt.py"],
        cwd=REPO_DIR,
        env={**os.environ, "DATA_PATH": "/mnt/data"},
        check=True,
    )

    # Persist logs to volume
    shutil.copytree(f"{REPO_DIR}/logs", "/mnt/data/logs", dirs_exist_ok=True)
