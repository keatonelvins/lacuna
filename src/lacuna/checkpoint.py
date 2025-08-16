"""Checkpoint saving and loading with torch.save."""

from typing import Any


def save_checkpoint(model: Any, optimizer: Any, step: int, path: str) -> None:
    """Save checkpoint."""
    pass


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load checkpoint."""
    pass