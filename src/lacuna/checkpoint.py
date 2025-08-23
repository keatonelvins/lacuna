"""Distributed checkpoint saving and loading."""

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from transformers import PreTrainedTokenizerBase
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
    """Stateful protocol for training metadata (step, tokens, MFU)."""

    def __init__(
        self,
        step: int = 0,
        total_tokens: int = 0,
        peak_mfu: float = 0.0,
        peak_tflops: float = 0.0,
    ):
        self.step = step
        self.total_tokens = total_tokens
        self.peak_mfu = peak_mfu
        self.peak_tflops = peak_tflops

    def state_dict(self) -> dict[str, Any]:
        """Get the training state dictionary."""
        return {
            "step": self.step,
            "total_tokens": self.total_tokens,
            "peak_mfu": self.peak_mfu,
            "peak_tflops": self.peak_tflops,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the training state."""
        self.step = state_dict["step"]
        self.total_tokens = state_dict["total_tokens"]
        self.peak_mfu = state_dict.get("peak_mfu", 0.0)
        self.peak_tflops = state_dict.get("peak_tflops", 0.0)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    total_tokens: int,
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    peak_mfu: float = 0.0,
    peak_tflops: float = 0.0,
    final: bool = False,
) -> None:
    """Save checkpoint in DCP format (intermediate) or HF format (final/single-GPU)."""
    if get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)

    # Get the actual model (unwrap from FSDP if needed)
    unwrapped_model = model.module if hasattr(model, "module") else model

    if not dist.is_initialized() or (final and get_rank() == 0):
        unwrapped_model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        logger.info(f"Saved HF model + tokenizer to {path}")

        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "total_tokens": total_tokens,
            "peak_mfu": peak_mfu,
            "peak_tflops": peak_tflops,
        }
        torch.save(training_state, path / "training_state.pt")
        logger.info(f"Saved training state to {path}/training_state.pt")

    else:
        state_dict = {
            "model": ModelState(model),
            "optimizer": OptimizerState(model, optimizer, scheduler),
            "training": TrainingState(step, total_tokens, peak_mfu, peak_tflops),
        }

        storage_writer = dcp.FileSystemWriter(str(path), overwrite=True)
        dcp.save(state_dict, storage_writer=storage_writer)
        logger.info(f"Saved DCP checkpoint to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    path: Path,
) -> dict[str, Any]:
    """Load checkpoint from either HF or DCP format."""
    is_hf_format = (path / "config.json").exists()
    is_dcp_format = (path / ".metadata").exists()

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    if not (is_hf_format or is_dcp_format):
        raise ValueError(f"Unknown checkpoint format at {path}")

    if is_hf_format:  # just load training state, model loaded from trainer
        training_state_path = path / "training_state.pt"
        training_state = torch.load(training_state_path, map_location="cpu")

        if "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler and training_state.get("scheduler"):
            scheduler.load_state_dict(training_state["scheduler"])

        logger.info(f"Loaded training state from {training_state_path}")

        return {
            "step": training_state.get("step", 0),
            "total_tokens": training_state.get("total_tokens", 0),
            "peak_mfu": training_state.get("peak_mfu", 0.0),
            "peak_tflops": training_state.get("peak_tflops", 0.0),
        }

    else:
        state_dict = {
            "model": ModelState(model),
            "optimizer": OptimizerState(model, optimizer, scheduler),
            "training": TrainingState(),
        }

        dcp.load(state_dict, checkpoint_id=str(path))
        logger.info(f"Loaded DCP checkpoint from {path}")

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
