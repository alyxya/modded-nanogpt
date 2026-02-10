Right after the script finishes, before you exit:

  1. Save the compiled kernel caches (saves 7 min next run)
  # if you have a Modal volume mounted at /vol
  cp -r ~/.triton/cache /vol/triton_cache
  cp -r ~/.cache/torch /vol/torch_cache

  2. Save the training log
  cp logs/*.txt /vol/

  3. Note your val_loss — it'll be in the last few lines of output. Jot it down or
  it's in the log file.

  That's it. The model checkpoint isn't saved by default (save_checkpoint: bool =
  False), so unless you changed that, there's nothing else to grab.

  Next run, restore the caches before training:
  cp -r /vol/triton_cache ~/.triton/cache
  cp -r /vol/torch_cache ~/.cache/torch

  To set this up, create a volume once:
  modal volume create nanogpt-vol

  Then launch with it mounted:
  modal shell --gpu H100:8 --volume nanogpt-vol:/vol
