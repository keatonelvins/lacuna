import os
import sys
import torch
import psutil
import tomllib
import itertools
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from torch.distributed.elastic.multiprocessing.errors import record

from lacuna.train import train as launch_train
from lacuna.config import TrainConfig, SweepConfig

load_dotenv()


def merge() -> None:
    """Alias for running against latest mergekit."""
    cmd = f"uv run --no-project --with git+https://github.com/arcee-ai/mergekit mergekit-yaml {' '.join(a for a in sys.argv[1:])}"
    os.execvp("bash", ["bash", "-c", cmd])


def chat() -> None:
    """Alias for running chat."""
    cmd = f"uv run scripts/chat.py {' '.join(a for a in sys.argv[1:])}"
    os.execvp("bash", ["bash", "-c", cmd])


@record
def train():
    """Main entry point for training loop."""
    config = parse_args()

    # if multi-gpu and haven't already launched torchrun, launch it
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        torchrun(config)
    else:
        launch_train(config)


def sweep():
    """Run consecutive trains, sweeping over the given args."""
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(SweepConfig.__doc__)
        exit(0)

    config_path, sweeps, fixed = parse_sweeps(args)
    sweep_keys, sweep_values = list(sweeps.keys()), list(sweeps.values())

    for combo in itertools.product(*sweep_values):
        sweep_overrides = [f"--{k}={v}" for k, v in zip(sweep_keys, combo)]
        fixed_overrides = [f"--{k}={v}" for k, v in fixed.items()]
        run_name = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(sweep_keys, combo))

        cmd = ["uv", "run", "train"]
        if config_path:
            cmd.append(config_path)
        cmd.extend(sweep_overrides + fixed_overrides + [f"--wandb.name={run_name}"])

        print(f"\n{'=' * 60}\nRunning: {run_name}\n{'=' * 60}")
        subprocess.run(cmd, check=True)


def slurm():
    """Submit job to slurm cluster."""
    args = sys.argv[1:]
    Path("logs").mkdir(exist_ok=True)
    config = parse_args()

    cmd = [
        "sbatch",
        f"-N {config.torchrun.nnodes}",
        f"--time={config.slurm.duration}:00:00",
        f"--job-name={config.trainer.run_name}",
    ]

    if config.slurm.queue:
        check_cmd = ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%N:%D"]
        result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)

        if result.stdout.strip():
            nodelist, nnodes = result.stdout.strip().split("\n")[0].split(":")
            if int(nnodes) != config.torchrun.nnodes:
                print(f"Error: Existing job has {nnodes} nodes but config specifies {config.torchrun.nnodes} nodes")
                sys.exit(1)

            print(f"Queueing on existing nodes: {nodelist}")
            cmd.extend(["--nodelist", nodelist])
        else:
            print("No jobs found, submitting without specific nodelist")

    cmd.extend(["scripts/slurm.sh"] + args)

    print(f"Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"Result: {result.stdout}\n{result.stderr}".strip())


def serve():
    """Serve model locally using sglang."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: uv run serve <model_path>")
        exit(0)

    cmd = f"uv run scripts/serve.py {' '.join(a for a in sys.argv[1:])}"
    os.execvp("bash", ["bash", "-c", cmd])


def torchrun(config: TrainConfig) -> None:
    """Relaunch command with torchrun and distributed flags."""
    if "OMP_NUM_THREADS" not in os.environ:
        cores = psutil.cpu_count(logical=False)
        os.environ["OMP_NUM_THREADS"] = str(max(1, cores // config.torchrun.nproc_per_node))

    cli_args = sys.argv[1:]
    if "--trainer.run_name" not in cli_args:
        cli_args.append(f"--trainer.run_name={config.trainer.run_name}")  # make sure all ranks have the same run name

    init_path = Path(__file__).resolve()
    cmd = ["torchrun", f"--nproc_per_node={config.torchrun.nproc_per_node}", str(init_path)] + cli_args

    if config.torchrun.node_rank is not None:  # set up multi-node
        cmd.insert(1, "--rdzv_backend=c10d")
        cmd.insert(1, f"--nnodes={config.torchrun.nnodes}")
        cmd.insert(1, f"--node_rank={config.torchrun.node_rank}")
        cmd.insert(1, f"--rdzv_endpoint={config.torchrun.master_addr}:{config.torchrun.master_port}")
        cmd.insert(1, f"--rdzv_id={config.torchrun.job_id}")
    elif config.torchrun.nnodes > 1:
        print(f"Error: For multi-node, (nnodes={config.torchrun.nnodes}) must specify node_rank")
        print("Example: uv run train configs/multi_node.toml --torchrun.node_rank 0")
        sys.exit(1)

    print(f"Launching: {' '.join(cmd)}")
    os.execvp("torchrun", cmd)  # execvp to replace current process


def parse_args() -> TrainConfig:
    """Parse cli args and load config from pydantic settings object."""
    args = sys.argv[1:]
    toml_data = {}

    # load overrides from toml file if provided
    if args and not args[0].startswith("--"):
        config_path, args = Path(args[0]), args[1:]
        toml_data = tomllib.load(open(config_path, "rb"))

    return TrainConfig(**toml_data, _cli_parse_args=args, _cli_implicit_flags=True)


def parse_sweep_args(value: str) -> list:
    """Parse sweep values (e.g. "1:10:1" -> [1, 2, ..., 10])."""
    if ":" in value:
        parts = value.split(":")
        start, stop = float(parts[0]), float(parts[1])
        step = float(parts[2]) if len(parts) > 2 else 1
        return [start + i * step for i in range(int((stop - start) / step) + 1)]
    return value.split(",")


def parse_sweeps(args: list[str]) -> tuple[str | None, dict[str, list], dict[str, str]]:
    """Parse sweeps and return config path, sweeped args, and fixed args."""
    if not args or args[0].startswith("--"):
        config_path, i = None, 0
    else:
        config_path, i = args[0], 1

    sweeps, fixed = {}, {}
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:]
            value = args[i + 1] if i + 1 < len(args) else ""
            if "," in value or ":" in value:
                sweeps[key] = parse_sweep_args(value)
            else:
                fixed[key] = value
            i += 2
        else:
            i += 1

    return config_path, sweeps, fixed


if __name__ == "__main__":
    train()  # entry point for torchrun
