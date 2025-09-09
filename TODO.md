# TODO

- Training Loop Perf
  - Non-blocking H2D: move batches with `tensor.cuda(non_blocking=True)` when `pin_memory=True`.
  - Prefetch/pipeline: add simple double-buffering (prefetch next batch to GPU on a separate stream).

- Data Pipeline
  - Better handle buffer (python list seems suboptimal)
  - need to call .shuffle() and use epochs()
  - Auto-tune `num_workers` based on CPU cores and dataset shard count; set `persistent_workers=True`; warn and clamp as needed (currently clamps to shards).
  - Strategy for poor sharding: guidance/reshard path when `dataset.num_shards != world_size`.

- Checkpointing
  - Switch to `dcp.async_save` per recipe; keep one in-flight save and await completion before starting another.
  - Add `dcp_to_hf` CLI in `cli.py` to repackage a DCP checkpoint into HF sharded weights.
  - Blocked on torch 2.9.0 (release 10/15):
      - Switch to `HuggingFaceStorageWriter` with `consolidate_safetensors_files_on_every_rank`

- Eval
  - Add eval loop with support for held-out split(s) of the same datasets.
  - Add support for verifiers envs as eval targets.

- Metrics
  - Improve FLOPs/MFU estimator (use AllenAI/veomni/prime-rl references); validate per-arch peak TFLOPs table.

- UX/Config
  - Validation: `sampling_probs` length matches `datasets`; save_dir writeable; tokenizer/model existence checks before training starts.
  - Figure out control surface for distributed: current dist and torchrun settings feel clunky, weird defaults now for single node HSDP

- Repro/Robustness
  - Global seeding: seed Python/Torch/CUDA and DataLoader workers.

- Optimizer/Model
  - need to handle weight-tying for small model sizes? (re: torchtitan, automodel)
  - Extract scheduler logic to `scheduler.py`; add prime-rl style schedulers while keeping current cosine/WSD.
  - Support FlexAttention backed with block-attention for masking (or SDPA if it supports varlen)

- MoE
  - Support grouped GEMM

- SFT
  - Support assistant-only loss