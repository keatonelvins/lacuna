# lacuna docs

## Attention Backends
| Mode | Backend | Fullgraph Compile | Notes |
|------|---------|------------------|-------|
| PT | SDPA |  Yes | Best performance w/ compile |
| PT | FA2 |  *Yes | *Can only use fullgraph if batch_size > 1 |
| PT | FA3 |  No | Using kernelhub impl. |
| SFT (no packing) | Any | *Yes | All backends work |
| SFT (packing) | FA2/FA3 |  No | SDPA blocked (needs varlen) |

## Order of Optimizations
Liger/CCE/Kernelize -> AC -> torch.compile -> FSDP
- Model patches always happens first
- Compile wrapped AC: compile already does recompute via the min-cut partitioner, so wrapping AC over a compiled region may lead to recomputing multiple times
- torch.compile before FSDP, otherwise FSDP2 wrapped modules would cause graph breaks

## Considerations
- Prefer regional (layer-wise) over full model compilation (https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)