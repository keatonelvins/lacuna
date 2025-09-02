"""lacuna cli entry point for uv scripts."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Type, TypeVar

import torch
import tomllib
import torch.distributed.checkpoint as dcp

from pydantic_settings import BaseSettings
from transformers import AutoTokenizer, AutoConfig
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from lacuna.config import PretrainConfig, SFTConfig
from lacuna.trainer import train

T = TypeVar("T", bound=BaseSettings)


def launch_torchrun(config: BaseSettings, entry_point: str) -> None:
    """Launch torchrun using config's torchrun settings."""
    torchrun = config.torchrun

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={torchrun.nproc_per_node}",
        f"--nnodes={torchrun.nnodes}",
        f"--master_addr={torchrun.master_addr}",
        f"--master_port={torchrun.master_port}",
    ]

    if hasattr(torchrun, "node_rank") and torchrun.node_rank is not None:
        cmd.append(f"--node_rank={torchrun.node_rank}")
    elif torchrun.nnodes > 1:
        print(
            f"Error: For multi-node training (nnodes={torchrun.nnodes}), node_rank must be specified in config or via --torchrun.node_rank"
        )
        print(
            "Example: uv run pt --torchrun configs/multi_node.toml --torchrun.node_rank 0"
        )
        sys.exit(1)

    cmd.extend(["-m", "lacuna.cli", entry_point])

    # Add original CLI args (without --torchrun flag)
    lacuna_args = [arg for arg in sys.argv[1:] if arg != "--torchrun"]
    cmd.extend(lacuna_args)

    print(f"Launching: {' '.join(cmd)}")

    # Use os.execvp due to better ctrl+c handling
    os.execvp("torchrun", cmd)


def parse_argv(config_cls: Type[T]) -> T:
    """Parse TOML config file and CLI overrides into pydantic settings"""
    args = sys.argv[1:]

    # First arg is TOML file path if it exists and doesn't start with --
    if args and not args[0].startswith("--"):
        config_path = Path(args[0])
        cli_args = args[1:]

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)
    else:
        toml_data = {}
        cli_args = args

    # Create config with TOML data as defaults, then apply CLI overrides
    config = config_cls(**toml_data, _cli_parse_args=cli_args)

    # Check for --torchrun flag after parsing config
    if "--torchrun" in args:
        entry_point = "pretrain_main" if config_cls == PretrainConfig else "sft_main"
        launch_torchrun(config, entry_point)

    return config


def pretrain():
    """Entry point for pretraining."""
    config = parse_argv(PretrainConfig)
    train(config)


def sft():
    """Entry point for SFT."""
    config = parse_argv(SFTConfig)
    train(config)


def dcp_to_hf():
    """Convert a lacuna DCP checkpoint (step dir) to HF sharded safetensors."""
    parser = argparse.ArgumentParser(
        description="Convert DCP step dir to HF sharded safetensors"
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to DCP step dir (contains model/.metadata)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: {checkpoint}_hf)"
    )
    args = parser.parse_args()

    src = args.checkpoint_path
    if not src.exists():
        print(f"Error: {src} does not exist")
        sys.exit(1)
    model_dir = src / "model"
    if not (model_dir / ".metadata").exists():
        print(f"Error: {src} must be a step directory containing model/.metadata")
        sys.exit(1)

    hf_path = args.output_dir if args.output_dir else src.parent / f"{src.name}_hf"
    hf_path.mkdir(parents=True, exist_ok=True)

    state_path = src / "training_state.json"
    if not state_path.exists():
        print(f"Error: {state_path} not found.")
        sys.exit(1)
    state = json.load(state_path.open("r"))
    model_name = state["hf_model_id"]

    tmp_pt = hf_path / "_weights.tmp.pt"
    dcp_to_torch_save(str(model_dir), str(tmp_pt))
    pt_state = torch.load(tmp_pt, map_location="cpu")
    tmp_pt.unlink(missing_ok=True)

    dcp.save(
        pt_state["model"],
        storage_writer=dcp.HuggingFaceStorageWriter(path=str(hf_path)),
        no_dist=True,
    )

    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(hf_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(hf_path)
