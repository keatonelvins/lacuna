# lacuna

Small, clean model training framework written for fun. With a hot hot tui!!

## Quick Start

```bash
# Install (or clone + `uv sync`) and test (requires ~5GB VRAM)
curl -sSL https://raw.githubusercontent.com/keatonelvins/lacuna/main/scripts/install.sh | bash

# See all config options
uv run train --help

# Load args from a toml (w/ temp overrides)
uv run train path/to/config.toml
uv run train path/to/config.toml --data.override_cache

# Pass args directly
uv run train --model.name Qwen/Qwen3-8B --trainer.steps 100

# Run a sweep (full grid)
uv run sweep path/to/config.toml --trainer.steps 10,20 --optimizer.lr 1e-5:5e-5:1e-5

# Run TUI
uv run lacuna
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
- [flame](https://github.com/fla-org/flame)
