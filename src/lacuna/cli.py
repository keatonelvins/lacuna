"""Entry point with TOML config loading and CLI overrides."""

import os
import sys
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
        "-m",
        "lacuna.cli",
        entry_point,
    ]

    # Add original CLI args (without --torchrun flag)
    lacuna_args = [arg for arg in sys.argv[1:] if arg != "--torchrun"]
    cmd.extend(lacuna_args)

    print(f"Launching: {' '.join(cmd)}")

    # Use os.execvp due to better ctrl+c handling
    os.execvp("torchrun", cmd)


def parse_argv(config_cls: Type[T], args: list[str] | None = None) -> T:
    """Parse TOML config file and CLI overrides into pydantic settings"""
    if args is None:
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


def pretrain_main():
    """Entry point for pretraining."""
    config = parse_argv(PretrainConfig)
    train(config)


def sft_main():
    """Entry point for SFT."""
    config = parse_argv(SFTConfig)
    train(config)


def dcp_to_hf_main():
    """Convert DCP checkpoint to HF format."""
    parser = argparse.ArgumentParser(
        description="Convert DCP checkpoint to HuggingFace format"
    )
    parser.add_argument("checkpoint_path", type=Path, help="Path to DCP checkpoint")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: {checkpoint}_hf)"
    )
    args = parser.parse_args()

    dcp_path = args.checkpoint_path
    if not dcp_path.exists() or not (dcp_path / ".metadata").exists():
        print(f"Error: {dcp_path} is not a valid DCP checkpoint")
        sys.exit(1)

    hf_path = (
        args.output_dir if args.output_dir else dcp_path.parent / f"{dcp_path.name}_hf"
    )
    hf_path.mkdir(parents=True, exist_ok=True)

    state_dict = {"config": {}}
    dcp.load(state_dict, checkpoint_id=str(dcp_path))
    model_name = state_dict["config"]["model"]["name"]

    weights_path = hf_path / "pytorch_model.bin"
    dcp_to_torch_save(str(dcp_path), str(weights_path))

    state_dict = torch.load(weights_path, map_location="cpu")
    torch.save(state_dict["model"], weights_path)

    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(hf_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(hf_path)
