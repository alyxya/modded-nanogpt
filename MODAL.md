# Modal Development Loops

This fork includes Modal entrypoints for running local experiments on remote GPUs without manually managing a GPU machine.

## Track 1

After authenticating Modal locally, run:

```bash
pip install modal
modal setup
modal run --detach modal_track1.py
```

By default this launches `train_gpt.py` on 1x H100 and caches FineWeb10B shards in a shared Modal volume, with Track 1-specific logs and compiler caches.

Common variants:

```bash
MODAL_GPU_TYPE=H100 MODAL_NUM_GPUS=8 modal run --detach modal_track1.py
modal run --detach modal_track1.py --num-data-shards 4
modal run --detach modal_track1.py --train-steps 60 --val-interval 25
modal run --detach modal_track1.py --train-steps 60 --extension-steps 1 --val-interval 25
```

## Track 3

Run the Track 3 optimization benchmark with:

```bash
pip install modal
modal setup
modal run --detach modal_track3.py
```

By default this launches the Track 3 benchmark on 1x H100 and uploads the current local working tree for that run. FineWeb10B shards are kept in a shared Modal volume, while logs and compiler caches use Track 3-specific volumes. The first run downloads data and later runs mostly just ship code changes.

Common variants:

```bash
# Use a different GPU shape. The benchmark supports 1, 2, 4, or 8 GPUs.
MODAL_GPU_TYPE=H100 MODAL_NUM_GPUS=2 modal run --detach modal_track3.py

# Use a scratch copy while experimenting.
modal run --detach modal_track3.py --script records/track_3_optimization/my_optimizer.py

# Download fewer train shards for a shortened scratch script, or more for longer runs.
modal run --detach modal_track3.py --num-data-shards 4
modal run --detach modal_track3.py --num-data-shards 100

# Override training steps and validation frequency, useful for cache warming or smoke tests.
modal run --detach modal_track3.py --train-steps 8 --val-interval 4
```
