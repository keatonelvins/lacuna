"""Distributed checkpoint saving and loading."""

from typing import Any
from loguru import logger
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
import torch.distributed as dist
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

from .config import LacunaConfig
from .data import get_tokenizer
from .distributed import is_master
from .utils import save_settings_json


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


# TODO: remove in torch 2.9.0 and use HuggingFaceStorageWriter
def save_hf_weights_dtensor(
    model: torch.nn.Module,
    output_dir: Path,
) -> None:
    sharded_sd = model.state_dict()
    cpu_state: dict[str, torch.Tensor] = {}

    for name, shard in sharded_sd.items():
        full_tensor = shard.full_tensor() if hasattr(shard, "full_tensor") else shard

        if is_master():
            cpu_state[name] = full_tensor.detach().cpu()
        else:
            del full_tensor

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])

    if is_master():
        model.save_pretrained(output_dir, state_dict=cpu_state)


def save_checkpoint(
    step: int,
    config: LacunaConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: StatefulDataLoader,
    final: bool = False,
) -> None:
    """Save DCP shards or final HF sharded weights."""
    ckpt_name = "final" if final else f"step_{step}"
    path = config.checkpoint.save_dir / ckpt_name

    unwrapped_model = model.module if hasattr(model, "module") else model
    unwrapped_model.config.save_pretrained(path)
    tokenizer = get_tokenizer(config)
    tokenizer.save_pretrained(path)

    if not final or config.checkpoint.resumable_final_save:
        logger.info(f"Saving resumable checkpoint to {path}")
        trainer_state = TrainerState(model, optimizer, scheduler, dataloader)
        writer = FileSystemWriter(str(path))
        state_dict = {"trainer": trainer_state}
        dcp.save(state_dict, storage_writer=writer)
    else:
        logger.info(f"Saving final checkpoint in HF format to {path}")
        save_hf_weights_dtensor(model, path)

    save_settings_json(path, config)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: StatefulDataLoader,
    path: Path,
):
    """Load DCP checkpoint and restore full training state."""

    if (path / "model.safetensors.index.json").exists():
        storage_reader = HuggingFaceStorageReader(path=str(path))
    elif (path / ".metadata").exists():
        storage_reader = FileSystemReader(str(path))
    else:
        raise ValueError(f"Checkpoint at {path} is neither DCP nor HF format")

    dcp.load(
        {"trainer": TrainerState(model, optimizer, scheduler, dataloader)},
        storage_reader=storage_reader,
        checkpoint_id=str(path),
    )

    logger.info(f"Loaded checkpoint from {path}")
