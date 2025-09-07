# TODO

- Usability
  - Auto-detect GPUs and torchrun: default `torchrun.nproc_per_node = torch.cuda.device_count()` and `nnodes=1`. If multi-GPU or any `--torchrun.*` arg is provided, re-exec under `torchrun` automatically; otherwise run single-process. Remove explicit `--torchrun` flag.
  - Hide torchrun/DDP noise: route logs through rank0 only (already using loguru); suppress torchrun banner/env chatter where possible.
  - Auto backend: if `world_size > 1` and `dist.backend == NONE`, default to `DDP`.

- Distributed/Runtime
  - Default torchrun when `torch.cuda.device_count()>1`; infer from args/world size (no explicit flag).
  - Default backend to DDP when `world_size>1` and backend is `NONE`.
  - DDP device binding: use `device_ids=[torch.cuda.current_device()]` (LOCAL_RANK) instead of global rank.
  - DDP buckets: expose `bucket_cap_mb` for better comm/compute overlap on small models.
  - Static graph: keep `static_graph=is_compiled`; document recommended `compile_mode` per backend.
  - Validate `DDP(static_graph=...)` vs `torch.compile` interaction; document recommended modes per backend.
  - Explore HSDP mesh (FSDP for blocks + DDP for data) and required wrapping order.

- Training Loop Perf
  - Non-blocking H2D: move batches with `tensor.cuda(non_blocking=True)` when `pin_memory=True`.
  - Prefetch/pipeline: add simple double-buffering (prefetch next batch to GPU on a separate stream).
  - Reset memory stats: call `torch.cuda.reset_peak_memory_stats()` at training start for cleaner metrics.

- Optimizer/Model
  - Fused AdamW param groups: keep `fused=True` and exclude weight decay on biases/LayerNorm and embeddings.
  - Extract scheduler logic to `scheduler.py`; add prime-rl style schedulers while keeping current cosine/WSD.

- Data Pipeline
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
  - Cleaner logs: suppress torchrun banner; keep rank0 logs only (already mostly done).

- Repro/Robustness
  - Global seeding: seed Python/Torch/CUDA and DataLoader workers.
