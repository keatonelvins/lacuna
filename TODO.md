# TODO

- Eval
  - Add eval loop with support for held-out split(s) of the same datasets.
  - Add support for verifiers envs as eval targets.

- UX/Config
  - Do dp_replicate/dp_shard instead of ddp/fsdp and additional hsdp flag

- Repro/Robustness
  - Global seeding: seed Python/Torch/CUDA and DataLoader workers.

- Metrics
  - Improve FLOPs/MFU estimator (use AllenAI/veomni/prime-rl references).

- SFT
  - Support assistant-only loss, load from chat template

- After v1: 
  - make a dataset config with name, split, etc so it's contained
  - look into hard map dataset caching with cache_file_name in .map()
  - Support grouped GEMM for MoE
  - Support FlexAttention backed with block-attention for masking (or SDPA if it supports varlen)
  - Switch to `dcp.async_save` per recipe; keep one in-flight save and await completion before starting another.
  - Add `dcp_to_hf` CLI in `cli.py` to repackage a DCP checkpoint into HF sharded weights.
  - Blocked on torch 2.9.0 (release 10/15):
      - Switch to `HuggingFaceStorageWriter` with `consolidate_safetensors_files_on_every_rank`
