# lacuna

Tiny, memory-efficient trainer for continued prelacunaing and finetuning of popular HF models.

Seeing how far we can push HF modeling code!

## Quick Start

```bash
# Install (or clone + `uv sync`) and test (requires ~5GB VRAM)
curl -sSL https://raw.githubusercontent.com/keatonelvins/lacuna/main/scripts/install.sh | bash
uv run lacuna

# See all config options
uv run lacuna --help

# Pass args directly
uv run lacuna --model.name Qwen/Qwen3-8B --trainer.batch-size 8

# Load args from a toml (w/ overrides)
uv run lacuna path/to/config.toml
uv run lacuna path/to/config.toml --optimizer.lr 1e-5

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
- [AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
