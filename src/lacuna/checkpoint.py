"""Distributed checkpoint saving and loading."""

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from loguru import logger

from .distributed import get_rank


class ModelState(Stateful):
    """Stateful protocol for model state."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        """Get the model's state dictionary with FSDP support."""
        return get_model_state_dict(
            self.model,
            options=StateDictOptions(cpu_offload=True),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dictionary into the model."""
        set_model_state_dict(self.model, state_dict)


class OptimizerState(Stateful):
    """Stateful protocol for optimizer and scheduler state."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        """Get the optimizer and scheduler state dictionaries."""
        optimizer_state_dict = get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(cpu_offload=True),
        )

        state_dict = {"optimizer": optimizer_state_dict}
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dictionaries into the optimizer and scheduler."""
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict["optimizer"],
        )

        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])


class TrainingState(Stateful):
    """Stateful protocol for training metadata (step, tokens)."""

    def __init__(self, step: int = 0, total_tokens: int = 0):
        self.step = step
        self.total_tokens = total_tokens

    def state_dict(self) -> dict[str, Any]:
        """Get the training state dictionary."""
        return {
            "step": self.step,
            "total_tokens": self.total_tokens,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the training state."""
        self.step = state_dict["step"]
        self.total_tokens = state_dict["total_tokens"]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    total_tokens: int,
    path: Path,
) -> None:
    """Save distributed checkpoint."""
    # Create parent directory if needed (only on rank 0)
    if get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "model": ModelState(model),
        "optimizer": OptimizerState(model, optimizer, scheduler),
        "training": TrainingState(step, total_tokens),
    }

    # Save using DCP (all ranks participate)
    dcp.save(state_dict, checkpoint_id=str(path))

    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    path: Path,
) -> dict[str, Any]:
    """Load distributed checkpoint."""

    if get_rank() == 0 and not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    state_dict = {
        "model": ModelState(model),
        "optimizer": OptimizerState(model, optimizer, scheduler),
        "training": TrainingState(),
    }

    dcp.load(state_dict, checkpoint_id=str(path))

    logger.info(f"Loaded distributed checkpoint from {path}")

    return state_dict["training"].state_dict()


def cleanup_old_checkpoints(save_dir: Path, keep_latest: int) -> None:
    """Cleanup old checkpoints, keeping the latest ones."""
    if get_rank() != 0:
        return

    if not save_dir.exists():
        return

    checkpoint_dirs = [d for d in save_dir.glob("step_*") if d.is_dir()]

    if len(checkpoint_dirs) <= keep_latest:
        return

    checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for old_checkpoint in checkpoint_dirs[keep_latest:]:
        shutil.rmtree(old_checkpoint)
        logger.info(f"Removed old checkpoint directory: {old_checkpoint}")
