# Lacuna Documentation

## Attention Backends

### Quick Reference
| Mode | Backend | Fullgraph Compile | Notes |
|------|---------|------------------|-------|
| PT | SDPA |  Yes | Best performance w/ compile |
| PT | FA2 |  Yes |  |
| PT | FA3 |  No | Using kernelhub impl. |
| SFT (no pack) | Any | Varies | All backends work |
| SFT (packing) | FA2/FA3 |  No | SDPA blocked (error) |

**Why no SDPA+packing?** SDPA doesn't respect position_id boundaries!
    - some examples floating around of SDPA with varlen attention, may look into eventually