# lacuna docs

My goal for `lacuna` is to boil down `torchtitan` into a more hf-friendly, minimal package:
- can be much simpler by not supporting TP/PP/CP
- no modeling code (for now), can lean on community kernels from kernel hub, liger, cce
- also support SFT w/ assistant-only loss and sample packing

I'm writing this mainly for personal use and because I learn best by building.

Personal goal: `uv run bloat_check` should return < 5k lines (currently clocking in ~2.5k)

## Attention backends
| Mode | Backend | Fullgraph Compile | Notes |
|------|---------|------------------|-------|
| PT | SDPA |  Yes | Best performance w/ compile |
| PT | FA3 |  No | Using kernelhub impl. |
| SFT (no packing) | Any | *Yes | All backends work |
| SFT (packing) | FA3 |  No | SDPA blocked (needs varlen) |

## Order of model builder
Liger/CCE/Kernelize -> AC -> torch.compile -> FSDP
- Model patches always happens first
- Compile wrapped AC: compile already does recompute via the min-cut partitioner, so wrapping AC over a compiled region may lead to recomputing multiple times
- torch.compile before FSDP, otherwise FSDP2 wrapped modules would cause graph breaks

## Other considerations
- Prefer regional (layer-wise) over full model compilation (https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- We default to fp32 accumulation (`accum_dtype=torch.float32`) for stability reasons (at the cost of some speed/memory):
    - For Liger Kernel, can pass through starting in `0.6.2`: https://github.com/linkedin/Liger-Kernel/pull/830
    - For CCE, we pass in `accum_e_fp32` and `accum_c_fp32`: https://github.com/axolotl-ai-cloud/ml-cross-entropy/blob/main/cut_cross_entropy/doc.py
