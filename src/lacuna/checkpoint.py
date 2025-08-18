"""Checkpoint saving and loading with torch.save."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from .distributed import get_rank


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    total_tokens: int,
    path: Path,
) -> None:
    """Save checkpoint with model, optimizer, and training state."""
    if get_rank() != 0:
        return

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint data
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "total_tokens": total_tokens,
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint from path."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location="cpu")
    logger.info(f"Loaded checkpoint from {path}")

    return checkpoint


def cleanup_old_checkpoints(save_dir: Path, keep_latest: int) -> None:
    """Remove old checkpoints, keeping only the most recent ones."""
    if get_rank() != 0:
        return

    if not save_dir.exists():
        return

    checkpoint_files = [f for f in save_dir.glob("step_*.pt") if f.is_file()]

    if len(checkpoint_files) <= keep_latest:
        return

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[keep_latest:]:
        old_checkpoint.unlink()
        logger.info(f"Removed old checkpoint: {old_checkpoint}")
