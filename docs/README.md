# Lacuna Documentation

## Attention Backends

### Compatibility
| Mode | Backend | Fullgraph Compile | Notes |
|------|---------|------------------|-------|
| PT | SDPA |  Yes | Best performance w/ compile |
| PT | FA2 |  *Yes | Can only use fullgraph if batch_size > 1 |
| PT | FA3 |  No | Using kernelhub impl. |
| SFT (no pack) | Any | Varies | All backends work |
| SFT (packing) | FA2/FA3 |  No | SDPA blocked (error) |

Packing needs varlen attention, have seen some examples using SDPA but will look into later

### Order of Optimizations
AC -> torch.compile
Today, compile wrapping AC is more recommended than AC wrapping compile. Compile already does recompute via the min-cut partitioner, so wrapping AC over a compiled region may lead to recomputing multiple times. If compile wraps AC, compile would incorporate the AC region information during the partitioner. It may be possible to improve this behavior though, e.g. detect any ambient AC contexts, and do something different in the partitioner and during runtime.

torch.compile -> FSDP
because torch.compile would graph break on FSDP2 wrapped modules.