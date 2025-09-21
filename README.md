# lacuna

Tiny, memory-efficient trainer for continued pretraining and finetuning of popular HF models.

## Quick Start

```bash
# Install (or clone + `uv sync`) and test (requires ~5GB VRAM)
curl -sSL https://raw.githubusercontent.com/keatonelvins/lacuna/main/scripts/install.sh | bash
uv run train

# See all config options
uv run train --help

# Load args from a toml (w/ temp overrides)
uv run train path/to/config.toml
uv run train path/to/config.toml --optimizer.lr 1e-5
uv run train path/to/config.toml --data.redownload

# Pass args directly
uv run train --model.name Qwen/Qwen3-8B --trainer.steps 100

# Run TUI
uv run lacuna

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

There are a ton of great training repos out there already! Some influences:
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
