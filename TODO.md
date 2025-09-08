# TODO

- Distributed/Runtime
  - Tune bucket_cap_mb for DDP base:
      scale = (12 * cfg.hidden_size**2) / 1e8
      bucket = 25 * (1 + scale)
      bucket *= 1.5 if world_size > 32 else 1
      return int(min(max(bucket, 10), 250))
  - Explore HSDP mesh (FSDP for blocks + DDP for data) and required wrapping order.

- Training Loop Perf
  - Non-blocking H2D: move batches with `tensor.cuda(non_blocking=True)` when `pin_memory=True`.
  - Prefetch/pipeline: add simple double-buffering (prefetch next batch to GPU on a separate stream).

- Optimizer/Model
  - Extract scheduler logic to `scheduler.py`; add prime-rl style schedulers while keeping current cosine/WSD.

- Data Pipeline
  - Better handle buffer (python list seems suboptimal)
  - Add SegmentTree-style packing to reduce truncation for pretraining streams.
  - Auto-tune `num_workers` based on CPU cores and dataset shard count; set `persistent_workers=True`; warn and clamp as needed (currently clamps to shards).
  - Strategy for poor sharding: guidance/reshard path when `dataset.num_shards != world_size`.

- Checkpointing
  - Switch to `dcp.async_save` per recipe; keep one in-flight save and await completion before starting another.
  - Clarify final save semantics: if `resumable_final_save=True`, final uses DCP (includes optimizer/scheduler/dataloader); else write HF sharded model-only. Document expected files.
  - Fix `load_checkpoint(...)`: support restoring full trainer state from DCP; gracefully handle HF model-only final checkpoints.
  - Add `dcp_to_hf` CLI in `cli.py` to repackage a DCP checkpoint into HF sharded weights.
  - Consider `consolidate_safetensors_files_on_every_rank` (torchtitan) to produce consolidated final shards.

- Eval
  - Add eval loop with support for held-out split(s) of the same datasets.
  - Add support for verifiers envs as eval targets.

- Metrics
  - Improve FLOPs/MFU estimator (use AllenAI/veomni/prime-rl references); validate per-arch peak TFLOPs table.

- UX/Config
  - Validation: `sampling_probs` length matches `datasets`; save_dir writeable; tokenizer/model existence checks before training starts.

- Repro/Robustness
  - Global seeding: seed Python/Torch/CUDA and DataLoader workers.

- MoE
  - Support grouped GEMM