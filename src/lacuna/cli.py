"""lacuna cli entry point."""

import os
import sys
import torch
import tomllib
from pathlib import Path
from dotenv import load_dotenv

from lacuna.config import LacunaConfig
from lacuna.trainer import train

load_dotenv()


def launch_torchrun(config: LacunaConfig) -> None:
    cmd = ["torchrun", f"--nproc_per_node={config.torchrun.nproc_per_node}"]

    if config.torchrun.node_rank is not None:
        cmd.extend(
            [
                f"--nnodes={config.torchrun.nnodes}",
                f"--master_addr={config.torchrun.master_addr}",
                f"--master_port={config.torchrun.master_port}",
                f"--node_rank={config.torchrun.node_rank}",
            ]
        )
    elif config.torchrun.nnodes > 1:
        print(f"Error: For multi-node training (nnodes={config.torchrun.nnodes}) must specify node_rank")
        print("Example: uv run train configs/multi_node.toml --torchrun.node_rank 0")
        sys.exit(1)

    cmd.extend(["-m", "lacuna.cli", "lacuna"] + sys.argv[1:])
    print(f"Launching: {' '.join(cmd)}")

    os.execvp("torchrun", cmd)


def parse_argv() -> LacunaConfig:
    args = sys.argv[1:]

    if args and not args[0].startswith("--"):
        config_path = Path(args[0])
        cli_args = args[1:]

        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)
    else:
        toml_data = {}
        cli_args = args

    config = LacunaConfig(**toml_data, _cli_parse_args=cli_args)

    # if multi-gpu and haven't already launched torchrun, launch it
    if torch.cuda.device_count() > 1 and "RANK" not in os.environ:
        launch_torchrun(config)

    return config


def lacuna():
    """Entry point for training."""
    config = parse_argv()
    train(config)


def main():
    """Torchrun entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run python -m lacuna.cli lacuna")
        sys.exit(1)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "lacuna":
        lacuna()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
