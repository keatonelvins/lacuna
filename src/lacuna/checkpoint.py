"""Distributed checkpoint saving and loading."""

from typing import Any
from loguru import logger
from pathlib import Path
import warnings

import torch
import torch.distributed.checkpoint as dcp
from transformers import PreTrainedTokenizerBase
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    HuggingFaceStorageReader,
    # HuggingFaceStorageWriter,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from .distributed import is_master
from .config import LacunaConfig
from .metrics import StateTracker
from .utils import save_state_json, save_settings_json, load_state_json


# ref: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
class TrainerState(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        dataloader: StatefulDataLoader,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer, options=StateDictOptions(cpu_offload=True)
        )
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict(),
            "dataloader": self.dataloader.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.dataloader.load_state_dict(state_dict["dataloader"])


def save_checkpoint(
    state: StateTracker,
    config: LacunaConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: StatefulDataLoader,
    tokenizer: PreTrainedTokenizerBase,
    final: bool = False,
) -> None:
    """Save DCP shards or final HF sharded weights."""
    ckpt_name = "final" if final else f"step_{state.step}"
    path = config.checkpoint.save_dir / ckpt_name
    if is_master():
        path.mkdir(parents=True, exist_ok=True)

    unwrapped_model = model.module if hasattr(model, "module") else model
    unwrapped_model.config.save_pretrained(path)
    tokenizer.save_pretrained(path)

    if not final or config.checkpoint.resumable_final_save:
        logger.info(f"Saving resumable checkpoint to {path}")
        trainer_state = TrainerState(model, optimizer, scheduler, dataloader)
        writer = FileSystemWriter(str(path))
        state_dict = {"trainer": trainer_state}
        with warnings.catch_warnings():  # ignore warnings if on single device
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
            dcp.save(state_dict, storage_writer=writer)
    else:
        logger.info(f"Saving final checkpoint in HF format to {path}")
        # TODO: use HuggingFaceStorageWriter in torch 2.9.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
            unwrapped_model.save_pretrained(path)

    save_state_json(path, state)
    save_settings_json(path, config)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: StatefulDataLoader,
    path: Path,
) -> StateTracker:
    """Load DCP checkpoint and restore full training state."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    is_dcp = (path / ".metadata").exists()
    is_hf = (path / "model.safetensors.index.json").exists()

    if is_hf:
        # TODO: remove and use HuggingFaceStorageReader in torch 2.9.0
        raise NotImplementedError("HF checkpoint loading not implemented")

    if not is_dcp and not is_hf:
        raise ValueError(f"Checkpoint at {path} is neither DCP nor HF format")

    if is_hf:
        storage_reader = HuggingFaceStorageReader(path=str(path))
    else:
        storage_reader = FileSystemReader(str(path))

    dcp.load(
        {"trainer": TrainerState(model, optimizer, scheduler, dataloader)},
        storage_reader=storage_reader,
        checkpoint_id=str(path),
    )

    logger.info(f"Loaded {'HF' if is_hf else 'DCP'} checkpoint from {path}")

    return load_state_json(path)
