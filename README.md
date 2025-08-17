# lacuna

## Quick Start

```bash
# Install
uv sync

# Pretraining
uv run pt configs/pretrain_qwen.toml
uv run pt configs/pretrain_qwen.toml --model.name Qwen/Qwen2.5-1.5B --max-steps 5000

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