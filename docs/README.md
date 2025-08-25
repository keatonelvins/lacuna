# Lacuna Documentation

## Attention Backends

### Quick Reference
| Mode | Backend | Fullgraph Compile | Notes |
|------|---------|------------------|-------|
| PT | SDPA |  Yes | Best performance |
| PT | FA2 |  Yes | No position_ids needed |
| PT | FA3 |  No | Kernelhub limitation |
| SFT (no pack) | Any | Varies | All backends work |
| SFT (packing) | FA2/FA3 |  No | SDPA blocked (error) |

### Configuration
```bash
uv run pt --model.attention SDPA   # Pretraining (optimal)
uv run sft --model.attention FA3   # SFT with packing
```

**Why no SDPA+packing?** SDPA doesn't respect position_id boundaries, 
causing incorrect cross-sequence attention in packed samples.