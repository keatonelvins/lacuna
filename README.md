# lacuna

Tiny, memory-efficient trainer for continued pretraining and finetuning of popular HF models

## Quick Start

```bash
# Single command install (w/ flash-attn)
uv sync

# See all config options
uv run pt --help
uv run sft --help

# Pass args directly
uv run pt --model.name Qwen/Qwen2.5-0.5B --max-steps 5000

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
```

## Acknowledgemnt

There are a ton of great training repos out there already! Some I have used/enjoyed:
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [ArticTraining](https://github.com/snowflakedb/ArcticTraining)
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- [trl](https://github.com/huggingface/trl/)