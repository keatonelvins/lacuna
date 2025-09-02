"""Distributed checkpoint saving and loading."""

import json
import shutil
from typing import Any
from loguru import logger
from pathlib import Path
from pydantic import BaseModel

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    FileSystemWriter,
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)
from torch.distributed.checkpoint.filesystem import SerializationFormat
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from transformers import PreTrainedTokenizerBase

from .distributed import get_rank
from .config import PretrainConfig, SFTConfig


class TrainingState(BaseModel):
    step: int = 0
    total_tokens: int = 0
    peak_mfu: float = 0.0
    peak_tflops: float = 0.0
    peak_mem_gb: float = 0.0


class ModelState(Stateful):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(
            self.model,
            options=StateDictOptions(cpu_offload=True),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(self.model, state_dict)


class OptimizerState(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        state_dict = {}
        state_dict["optimizer"] = get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(cpu_offload=True),
        )
        state_dict["scheduler"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict["optimizer"],
        )

        self.scheduler.load_state_dict(state_dict["scheduler"])


def _write_training_state_json(
    path: Path,
    training_state: TrainingState,
    config: PretrainConfig | SFTConfig,
) -> None:
    if get_rank() != 0:
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "training_state.json").open("w") as f:
        json.dump(training_state.model_dump(), f, indent=4)
    with (path / "settings.json").open("w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=4)


def _read_training_state_json(path: Path) -> TrainingState:
    ts_path = path / "training_state.json"
    if ts_path.exists():
        with ts_path.open("r") as f:
            return TrainingState(**json.load(f))
    return TrainingState()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    config: PretrainConfig | SFTConfig,
    state: TrainingState,
    final: bool = False,
) -> None:
    """Save DCP shards or final HF sharded weights."""
    if get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)

    unwrapped_model = model.module if hasattr(model, "module") else model

    if not final:
        model_dir = path / "model"
        optim_dir = path / "optim"

        dcp.save(
            {"model": ModelState(model)},
            storage_writer=FileSystemWriter(
                str(model_dir), serialization_format=SerializationFormat.SAFETENSORS
            ),
        )
        dcp.save(
            {"optim": OptimizerState(model, optimizer, scheduler)},
            storage_writer=FileSystemWriter(
                str(optim_dir), serialization_format=SerializationFormat.SAFETENSORS
            ),
        )
        _write_training_state_json(path, state, config)
        logger.info(f"Saved DCP checkpoint shards to {path}")
        return

    weights_sd = ModelState(model).state_dict()
    dcp.save(weights_sd, storage_writer=HuggingFaceStorageWriter(path=str(path)))

    if get_rank() == 0:
        unwrapped_model.config.save_pretrained(path)
        tokenizer.save_pretrained(path)

    if config.checkpoint.resumable_final_save:
        optim_dir = path / "optim"
        dcp.save(
            {"optim": OptimizerState(model, optimizer, scheduler)},
            storage_writer=FileSystemWriter(
                str(optim_dir), serialization_format=SerializationFormat.SAFETENSORS
            ),
        )
    _write_training_state_json(path, state, config)

    logger.info(f"Saved final HF sharded checkpoint to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    path: Path,
) -> TrainingState:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    is_step_dir = (path / "model" / ".metadata").exists() and (
        path / "optim" / ".metadata"
    ).exists()
    is_hf_final = (path / "model.safetensors.index.json").exists()

    if is_step_dir:
        dcp.load({"model": ModelState(model)}, checkpoint_id=str(path / "model"))
        dcp.load(
            {"optim": OptimizerState(model, optimizer, scheduler)},
            checkpoint_id=str(path / "optim"),
        )
        logger.info(f"Loaded DCP checkpoint from {path}")
        return _read_training_state_json(path)

    elif is_hf_final:
        optim_meta = (path / "optim" / ".metadata").exists()
        ts_json = (path / "training_state.json").exists()
        if not (optim_meta and ts_json):
            raise ValueError(
                f"Final checkpoint at {path} is not resumable. Set resumable_final_save=True during save."
            )
        dcp.load(
            {"model": ModelState(model)},
            storage_reader=HuggingFaceStorageReader(path=str(path)),
        )
        dcp.load(
            {"optim": OptimizerState(model, optimizer, scheduler)},
            checkpoint_id=str(path / "optim"),
        )
        logger.info(f"Loaded HF final (resumable) checkpoint from {path}")
        return _read_training_state_json(path)

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
