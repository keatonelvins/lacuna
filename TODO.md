# TODO

- Eval
  - Add eval loop with support for held-out split(s) of the same datasets.
  - Add support for verifiers envs as eval targets.

- After v1: 
  - Try out spawn for torch mp and create separate tokenizer instances per dataloader worker
  - bump liger kernel for DFT loss: https://arxiv.org/pdf/2508.05629
  - Improve FLOPs/MFU estimator (use AllenAI/veomni/prime-rl references).
  - Make a dataset config with name, split, etc so it's contained
  - Look into hard map dataset caching with cache_file_name in .map()
  - Support grouped GEMM for MoE(?), options are: kernels hub, torchtitan dependency/axolotl moe-kernels-v2
  - Switch to `dcp.async_save` per recipe; keep one in-flight save and await completion before starting another.
  - Add `dcp_to_hf` CLI in `cli.py` to repackage a DCP checkpoint into HF sharded weights.
  - Tune ugly `pack_bfd`, maybe rewrite in torch and see if performance hit is tolerable
  - Blocked on torch 2.9.0 (release 10/15):
      - Switch to `HuggingFaceStorageWriter` with `consolidate_safetensors_files_on_every_rank`
      - Allow SDPA w/ varlen option for attention backend (or also try FlexAttention w/ block mask)