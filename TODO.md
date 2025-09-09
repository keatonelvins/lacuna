# TODO

- Data Pipeline
  - Look into how split_dataset_by_node and `dp_replicate` and `dp_shard` interact
  - standardize around epochs

- Eval
  - Add eval loop with support for held-out split(s) of the same datasets.
  - Add support for verifiers envs as eval targets.

- Metrics
  - Improve FLOPs/MFU estimator (use AllenAI/veomni/prime-rl references); validate per-arch peak TFLOPs table.

- UX/Config
  - Figure out control surface for distributed: current dist and torchrun settings feel clunky, weird defaults now for single node HSDP

- Repro/Robustness
  - Global seeding: seed Python/Torch/CUDA and DataLoader workers.

- Optimizer/Model
  - need to handle weight-tying for small model sizes? (re: torchtitan, automodel)
  - Extract scheduler logic to `scheduler.py`; add prime-rl style schedulers while keeping current cosine/WSD.
  - Support FlexAttention backed with block-attention for masking (or SDPA if it supports varlen)

- Checkpointing
  - Switch to `dcp.async_save` per recipe; keep one in-flight save and await completion before starting another.
  - Add `dcp_to_hf` CLI in `cli.py` to repackage a DCP checkpoint into HF sharded weights.
  - Blocked on torch 2.9.0 (release 10/15):
      - Switch to `HuggingFaceStorageWriter` with `consolidate_safetensors_files_on_every_rank`

- MoE
  - Support grouped GEMM

- SFT
  - Support assistant-only loss