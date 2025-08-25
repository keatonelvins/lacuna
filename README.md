# lacuna

Tiny, memory-efficient trainer for continued pretraining and finetuning of popular HF models.

Seeing how far we can push HF modeling code!

## Quick Start

```bash
# Single command install (or clone + `uv sync`)
curl -sSL https://raw.githubusercontent.com/keatonelvins/lacuna/main/scripts/install.sh | bash

# See all config options
uv run pt --help
uv run sft --help

# Pass args directly
uv run pt --model.name Qwen/Qwen2.5-0.5B --trainer.steps 5000

# Load args from a toml (w/ overrides)
uv run sft path/to/config.toml
uv run sft path/to/config.toml --optimizer.lr 1e-5

# Multi-GPU w/ torchrun (multi-node otw)
uv run pt --torchrun path/to/config.toml

# Convert an early checkpoint to hf safetensors
uv run dcp_to_hf --help

# Run benchmarks for different speedups
uv run benchmark

# Log runs + access gated repos
uv run wandb login
uv run hf auth login
```

## Development

```bash
uv sync --dev
uv run pre-commit install
```

## Acknowledgemnt

There are a ton of great training repos out there already! Some that influenced my implementation:
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)


## Notes

- We default to fp32 for gradient accumulation (`accum_dtype=torch.float32`) for stability reasons (at the cost of some speed/memory):
    - For LigerKernel, can pass through starting in `0.6.2`: https://github.com/linkedin/Liger-Kernel/pull/830
    - For CCE, we pass in `accum_e_fp32` and `accum_c_fp32`: https://github.com/axolotl-ai-cloud/ml-cross-entropy/blob/main/cut_cross_entropy/doc.py
- FA2 and torch.compile(fullgraph=True) can't be stacked when batch_size = 1 due to HF limitation.
