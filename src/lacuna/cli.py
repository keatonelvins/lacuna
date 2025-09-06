"""lacuna cli entry point for uv scripts."""

import os
import sys
import subprocess
import tomllib
from pathlib import Path
from typing import Type, TypeVar
from pydantic_settings import BaseSettings

from lacuna.config import PretrainConfig, SFTConfig
from lacuna.trainer import train

T = TypeVar("T", bound=BaseSettings)


def launch_torchrun(config: BaseSettings, entry_point: str) -> None:
    """Launch torchrun using config's torchrun settings."""
    torchrun = config.torchrun

    cmd = [
        "torchrun",
        f"--nproc_per_node={torchrun.nproc_per_node}",
        f"--nnodes={torchrun.nnodes}",
        f"--master_addr={torchrun.master_addr}",
        f"--master_port={torchrun.master_port}",
    ]

    if torchrun.node_rank is not None:
        cmd.append(f"--node_rank={torchrun.node_rank}")
    elif torchrun.nnodes > 1:
        print(f"Error: For multi-node training (nnodes={torchrun.nnodes}) must specify node_rank")
        print("Example: uv run pt --torchrun configs/multi_node.toml --torchrun.node_rank 0")
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


def count_lines():
    """Check the total repo line count."""
    result = subprocess.run(
        "git ls-files '*.py' | xargs cat | wc -l",
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    line_count = int(result.stdout.strip())
    print(f"Total repo lines: {line_count}")
