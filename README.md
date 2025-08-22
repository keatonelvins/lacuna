# lacuna

Tiny, memory-efficient trainer for continued pretraining and finetuning of popular HF models

## Quick Start

```bash
# Single command install (w/ FA2)
uv sync

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

# Log runs + access gated repos
uv run wandb login
uv run hf auth login
```

## Development

```bash
uv sync --dev
uv run pre-commit install

# Bump dependencies
uv lock --upgrade && uv sync
```

## Acknowledgemnt

There are a ton of great training repos out there already! Some I have used/enjoyed:
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [ArticTraining](https://github.com/snowflakedb/ArcticTraining)
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- [trl](https://github.com/huggingface/trl/)

## Notes

- We default to fp32 for gradient accumulation (`accum_dtype=torch.float32`) for stability reasons (at the cost of some speed/memory):
    - For LigerKernel, can pass through starting in `0.6.2`: https://github.com/linkedin/Liger-Kernel/pull/830
    - For CCE, we pass in `accum_e_fp32` and `accum_c_fp32`: https://github.com/axolotl-ai-cloud/ml-cross-entropy/blob/main/cut_cross_entropy/doc.py