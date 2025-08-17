# lacuna

Tiny, memory-efficient trainer for continued pretraining and finetuning of popular HF models

## Quick Start

```bash
# Install
uv sync

# Pretraining
uv run pt configs/pt_qwen.toml
uv run pt configs/pt_qwen.toml --model.name Qwen/Qwen2.5-1.5B --max-steps 5000

# Fine-tuning  
uv run sft configs/sft_qwen.toml
uv run sft configs/sft_qwen.toml --optimizer.lr 1e-5 --epochs 3

# See all options
uv run pt --help
uv run sft --help

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