"""Distributed checkpoint saving and loading."""

from typing import Any
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from lacuna.config import LacunaConfig
from lacuna.data import get_tokenizer
from lacuna.distributed import is_master
from lacuna.utils import save_settings


class TrainerState(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        dataloader: StatefulDataLoader | None = None,
        step: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.step = step

    def state_dict(self) -> dict[str, Any]:
        state_dict = {"step": self.step}
        state_dict["model"], state_dict["optimizer"] = get_state_dict(
            self.model, self.optimizer, options=StateDictOptions(cpu_offload=True)
        )

        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self.dataloader is not None:
            state_dict["dataloader"] = self.dataloader.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict.get("step", 0)

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )

        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if "dataloader" in state_dict and self.dataloader is not None:
            self.dataloader.load_state_dict(state_dict["dataloader"])


def save_checkpoint(
    step: int,
    config: LacunaConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataloader: StatefulDataLoader | None,
) -> None:
    """Save DCP checkpoint. Pass None to exclude components."""
    path = config.checkpoint.save_dir / f"step_{step}"
    logger.info(f"Saving checkpoint to {path}")

    trainer_state = TrainerState(model, optimizer, scheduler, dataloader, step)
    dcp.save({"trainer": trainer_state}, checkpoint_id=str(path))

    if is_master():
        get_tokenizer(config).save_pretrained(path)
        model.config.save_pretrained(path)
        save_settings(path, config)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataloader: StatefulDataLoader | None,
    path: Path,
) -> int:
    """Load DCP checkpoint. Pass None to skip components. Returns step."""
    if not (path / ".metadata").exists():
        raise ValueError(f"Not a DCP checkpoint: {path}")

    trainer_state = TrainerState(model, optimizer, scheduler, dataloader)
    dcp.load({"trainer": trainer_state}, checkpoint_id=str(path))
    return trainer_state.step
