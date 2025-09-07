"""lacuna cli entry point for uv scripts."""

import os
import sys
import subprocess
import tomllib
from pathlib import Path
from typing import Type, TypeVar
from pydantic_settings import BaseSettings

import torch

from lacuna.config import PretrainConfig, SFTConfig
from lacuna.trainer import train

T = TypeVar("T", bound=BaseSettings)


def launch_torchrun(config: BaseSettings, entry_point: str) -> None:
    torchrun = config.torchrun

    cmd = ["torchrun", f"--nproc_per_node={torchrun.nproc_per_node}"]

    if torchrun.node_rank is not None:
        cmd.extend([
            f"--nnodes={torchrun.nnodes}",
            f"--master_addr={torchrun.master_addr}",
            f"--master_port={torchrun.master_port}",
            f"--node_rank={torchrun.node_rank}"
        ])
    elif torchrun.nnodes > 1:
        print(f"Error: For multi-node training (nnodes={torchrun.nnodes}) must specify node_rank")
        print("Example: uv run pt configs/multi_node.toml --torchrun.node_rank 0")
        sys.exit(1)

    cmd.extend(["-m", "lacuna.cli", entry_point])
    cmd.extend(sys.argv[1:])

    print(f"Launching: {' '.join(cmd)}")

    os.execvp("torchrun", cmd)


def parse_argv(config_cls: Type[T]) -> T:
    args = sys.argv[1:]

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

    config = config_cls(**toml_data, _cli_parse_args=cli_args)

    # if we are not in a distributed environment, launch torchrun
    if torch.cuda.device_count() > 1 and "RANK" not in os.environ:
        entry_point = "pretrain" if config_cls == PretrainConfig else "sft"
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


def main():
    """Torchrun entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run python -m lacuna.cli [pretrain|sft]")
        sys.exit(1)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "pretrain":
        pretrain()
    elif cmd == "sft":
        sft()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
