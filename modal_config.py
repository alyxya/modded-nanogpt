import modal

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

@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={
        "/mnt/data": modal.Volume.from_name("nanogpt-data", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.inductor-cache": modal.Volume.from_name("inductor-cache", create_if_missing=True),
    },
)
def dev():
    """Shell target for interactive development."""
    pass
