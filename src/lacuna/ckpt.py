"""Distributed checkpoint saving and loading."""

import re
import json
import tomli_w
import shutil
from typing import Any
from pathlib import Path
from loguru import logger

import torch
from transformers import GenerationConfig
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import HuggingFaceStorageReader, HuggingFaceStorageWriter, DefaultLoadPlanner
from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files_on_every_rank
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from huggingface_hub import hf_hub_download, snapshot_download

from lacuna.config import TrainConfig
from lacuna.utils import is_master, run_master_first
from lacuna.scripts.adapters import convert
from lacuna.data import get_tokenizer


# ref: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class TrainerState(Stateful):
    """Stateful tracker for saving/loading checkpoints."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        dataloader: StatefulDataLoader | None = None,
        step: int = 0,
    ):
        self.step = step
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader

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


def get_safetensors_index(model_name: str) -> dict[str, int] | None:
    """Get the `model.safetensors.index.json` from local cache or from HF hub."""
    try:
        with run_master_first():
            if Path(model_name).exists():
                index_path = Path(model_name) / "model.safetensors.index.json"
            else:
                index_path = Path(hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json"))

        index_data = json.load(index_path.open())

        fqn_to_index_mapping = {}
        for fqn, filename in index_data.get("weight_map", {}).items():
            match = re.search(r"(\d+)", filename)  # 'model-00004-of-00013.safetensors' -> '00004'
            file_idx = int(match.group(0)) if match else 1  # '00004' -> 4
            fqn_to_index_mapping[fqn] = file_idx

        return fqn_to_index_mapping, index_data
    except Exception:
        logger.info("Unable to locate model.safetensors.index.json, dumping to single file")
        return None, None


def clean_save_dir(config: TrainConfig) -> None:
    """Remove old checkpoints, keeping only the latest config.ckpt.saves."""
    save_dir = config.ckpt.save_dir
    if not save_dir or not save_dir.exists():
        return

    discovered = []
    for path in save_dir.iterdir():
        if match := re.search(r"step_(\d+)", path.name):
            discovered.append((int(match.group(1)), path))

    discovered.sort()
    to_delete = discovered[: -1 * config.ckpt.saves]

    for _, path in to_delete:
        logger.info(f"Deleting old save: {path}")
        shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(
    step: int,
    resumable: bool,
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataloader: StatefulDataLoader | None,
    final: bool = False,
) -> None:
    """Save checkpoint: resumable DCP or model-only safetensors."""
    path = config.ckpt.save_dir / f"step_{step}"
    logger.info(f"Saving {'resumable' if resumable else 'model-only'} checkpoint to {path}")

    if resumable:
        trainer_state = TrainerState(model, optimizer, scheduler, dataloader, step)
        dcp.save({"trainer": trainer_state}, checkpoint_id=str(path))
    else:
        model_state_dict, _ = get_state_dict(model, optimizer, options=StateDictOptions(cpu_offload=True))
        fqn_to_index_mapping, index_data = get_safetensors_index(config.model.name)

        # if safetensors index found, first save shards then consolidate to hf format. if not, dump to single .safetensors file
        if fqn_to_index_mapping:
            storage_writer = HuggingFaceStorageWriter(
                path=str(path / "sharded"),
                fqn_to_index_mapping=fqn_to_index_mapping,
                save_distributed=True,
                enable_consolidation=False,
            )
            dcp.save(model_state_dict, storage_writer=storage_writer)
            consolidate_safetensors_files_on_every_rank(
                input_dir=str(path / "sharded"),
                output_dir=str(path),
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )
            if is_master():
                json.dump(index_data, open(path / "model.safetensors.index.json", "w"))
        else:
            storage_writer = HuggingFaceStorageWriter(path=str(path), save_distributed=True, enable_consolidation=True)
            dcp.save(model_state_dict, storage_writer=storage_writer)

        if is_master():
            shutil.rmtree(path / "sharded", ignore_errors=True)

    if is_master():
        clean_save_dir(config)
        tokenizer = get_tokenizer(config)
        generation_config = GenerationConfig.from_pretrained(config.model.name)
        generation_config.eos_token_id = tokenizer.eos_token_id  # in case we added a new eos token

        tokenizer.save_pretrained(path)
        model.config.save_pretrained(path)
        generation_config.save_pretrained(path)
        with open(path / "settings.toml", "wb") as f:
            tomli_w.dump(config.model_dump(exclude_defaults=True, mode="json"), f)

        if final and config.model.lacuna:  # convert tt moe -> hf for final checkpoint
            convert(str(config.ckpt.save_dir / f"step_{step}"), None, to_hf=True)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataloader: StatefulDataLoader | None,
) -> int:
    """Load trainer state from DCP checkpoint (scheduler and dataloader can be skipped)."""
    trainer_state = TrainerState(model, optimizer, scheduler, dataloader)
    dcp.load({"trainer": trainer_state}, checkpoint_id=str(path))
    return trainer_state.step


def load_pretrained_weights(model: torch.nn.Module, model_path: str) -> None:
    """Load pretrained HF weights after FSDP wrapping via DCP."""
    logger.info(f"Loading pretrained weights from {model_path}")
    if not Path(model_path).exists():
        model_path = snapshot_download(repo_id=model_path, repo_type="model")

    dcp.load(
        model.state_dict(),
        storage_reader=HuggingFaceStorageReader(path=model_path),
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    logger.info("Pretrained weights loaded successfully")
