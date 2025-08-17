# lacuna

### Setup
```bash
uv sync && uv sync --extra flash-attn
uv run wandb login
uv run hf auth login
```

### Development
```bash
uv sync --dev
uv run pre-commit install
```