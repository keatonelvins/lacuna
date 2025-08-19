"""Entry point with TOML config loading and CLI overrides."""

import os
import sys
from pathlib import Path
from typing import Type, TypeVar


import tomllib
from pydantic_settings import BaseSettings

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
