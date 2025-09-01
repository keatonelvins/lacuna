# lacuna docs

My goal is for this to be (in spirit) a nano-torchtitan with the following modifications:
- can be much simpler by not supporting TP/PP/CP
- no modeling code (for now) means we can lean on community resources like Liger Kernel and CCE
- also support SFT w/ assistant-only loss and sample packing

Ethos is most similar to prime-rl, but focused on PT/SFT rather than SFT/RL.

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
    - For LigerKernel, can pass through starting in `0.6.2`: https://github.com/linkedin/Liger-Kernel/pull/830
    - For CCE, we pass in `accum_e_fp32` and `accum_c_fp32`: https://github.com/axolotl-ai-cloud/ml-cross-entropy/blob/main/cut_cross_entropy/doc.py
