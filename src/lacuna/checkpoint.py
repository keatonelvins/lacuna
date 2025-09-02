"""Distributed checkpoint saving and loading."""

import shutil
from typing import Any
from loguru import logger
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import PreTrainedTokenizerBase
from torch.distributed.checkpoint import (
    FileSystemWriter,
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)
from torch.distributed.checkpoint.filesystem import SerializationFormat
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from .distributed import get_rank
from .config import PretrainConfig, SFTConfig
from .metrics import StateTracker
from .utils import save_state_json, save_settings_json, load_state_json


class TrainerState(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        dataloader: StatefulDataLoader | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer, options=StateDictOptions(cpu_offload=True)
        )
        scheduler_state_dict = self.scheduler.state_dict()
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
        }
        if self.dataloader is not None:
            state_dict["dataloader"] = self.dataloader.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.dataloader is not None:
            assert "dataloader" in state_dict
            self.dataloader.load_state_dict(state_dict["dataloader"])


def save_checkpoint(
    path: Path,
    state: StateTracker,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: StatefulDataLoader,
    config: PretrainConfig | SFTConfig,
    tokenizer: PreTrainedTokenizerBase,
    final: bool = False,
) -> None:
    """Save DCP shards or final HF sharded weights."""
    if get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)

    trainer_state = TrainerState(model, optimizer, scheduler, dataloader)
    if not final or config.checkpoint.resumable_final_save:
        writer = FileSystemWriter(
            str(path), serialization_format=SerializationFormat.SAFETENSORS
        )
    else:
        writer = HuggingFaceStorageWriter(path=str(path))
        unwrapped_model = model.module if hasattr(model, "module") else model
        unwrapped_model.config.save_pretrained(path)
        tokenizer.save_pretrained(path)
    dcp.save({"trainer": trainer_state}, storage_writer=writer)

    save_state_json(path, state)
    save_settings_json(path, config)
    logger.info(f"Saved DCP checkpoint shards to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    path: Path,
) -> StateTracker:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    is_resumable = (path / ".metadata").exists()
    is_hf_final = (path / "model.safetensors.index.json").exists()

    if is_resumable:
        dcp.load(
            {"trainer": TrainerState(model, optimizer, scheduler)},
            checkpoint_id=str(path / "trainer"),
        )
        logger.info(f"Loaded DCP checkpoint from {path}")
        return load_state_json(path)
    elif is_hf_final:
        dcp.load(
            {"trainer": TrainerState(model, optimizer, scheduler)},
            storage_reader=HuggingFaceStorageReader(path=str(path)),
        )
        logger.info(f"Loaded HF final (resumable) checkpoint from {path}")
        return load_state_json(path)

    raise ValueError(f"Unknown checkpoint format at {path}")


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
