"""lacuna cli entry point."""

import os
import sys
import torch
import tomllib
import subprocess
import itertools
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

    cmd.extend(["-m", "lacuna.cli"] + sys.argv[1:])
    print(f"Launching: {' '.join(cmd)}")

    os.execvp("torchrun", cmd)


def parse_argv() -> LacunaConfig:
    args = sys.argv[1:]

    if args and not args[0].startswith("--"):  # load from toml
        config_path = Path(args[0])
        cli_args = args[1:]

        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)
    else:
        toml_data = {}
        cli_args = args

    config = LacunaConfig(**toml_data, _cli_parse_args=cli_args, _cli_implicit_flags=True)

    # if multi-gpu and haven't already launched torchrun, launch it
    if torch.cuda.device_count() > 1 and "RANK" not in os.environ:
        launch_torchrun(config)

    return config


def parse_sweep_value(value: str) -> list:
    if ":" in value:
        parts = value.split(":")
        start, stop = float(parts[0]), float(parts[1])
        step = float(parts[2]) if len(parts) > 2 else 1
        return [start + i * step for i in range(int((stop - start) / step) + 1)]
    return value.split(",")


def parse_sweep_args(args: list[str]) -> tuple[str | None, dict[str, list], dict[str, str]]:
    config_path = None if not args or args[0].startswith("--") else args[0]
    sweeps = {}
    fixed = {}

    i = 0 if config_path is None else 1
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:]
            value = args[i + 1] if i + 1 < len(args) else ""
            if "," in value or ":" in value:
                sweeps[key] = parse_sweep_value(value)
            else:
                fixed[key] = value
            i += 2
        else:
            i += 1

    return config_path, sweeps, fixed


def run_sweeps(config_path: str | None, sweeps: dict[str, list], fixed: dict[str, str]) -> None:
    keys, values = list(sweeps.keys()), list(sweeps.values())

    for combo in itertools.product(*values):
        sweep_overrides = [f"--{k}={v}" for k, v in zip(keys, combo)]
        fixed_overrides = [f"--{k}={v}" for k, v in fixed.items()]
        run_name = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo))

        cmd = ["uv", "run", "train"]
        if config_path:
            cmd.append(config_path)
        cmd.extend(sweep_overrides + fixed_overrides + [f"--wandb.name={run_name}"])

        print(f"\n{'=' * 60}\nRunning: {run_name}\n{'=' * 60}")
        subprocess.run(cmd, check=True)


def sweep():
    config_path, sweeps, fixed = parse_sweep_args(sys.argv[1:])
    run_sweeps(config_path, sweeps, fixed)


def run_train():
    """Main entry point for training loop."""
    train(parse_argv())


if __name__ == "__main__":
    run_train()
