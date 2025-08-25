"""Entry point with TOML config loading and CLI overrides."""

import os
import sys
import argparse
import json
import subprocess
import time
import gc
from datetime import datetime
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


def benchmark_main():
    """Entry point for benchmark."""
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset of benchmarks to run (default: 'all')",
    )
    args = parser.parse_args()

    benchmarks_dir = Path("configs/benchmarks")
    benchmark_configs = sorted(benchmarks_dir.glob("*.toml"))
    if args.subset != "all":
        benchmark_configs = [
            config for config in benchmark_configs if args.subset in config.stem
        ]

    results_dir = Path("benchmark_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running {len(benchmark_configs)} benchmarks")
    print(f"{'=' * 60}\n")

    results = []
    start_time = time.perf_counter()

    for i, config_path in enumerate(benchmark_configs, 1):
        config_name = config_path.stem
        print(f"[{i}/{len(benchmark_configs)}] Running {config_name}...")

        # Get save_dir from config
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
        save_dir = Path(
            config_data.get("checkpoint", {}).get(
                "save_dir", f"benchmark_results/{config_name}"
            )
        )

        # Run benchmark
        run_start = time.perf_counter()
        try:
            process = subprocess.run(
                ["uv", "run", "pt", str(config_path)],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            runtime = time.perf_counter() - run_start

            # Always check for checkpoint, regardless of exit code
            training_state_path = save_dir / "final" / "training_state.pt"
            if training_state_path.exists():
                try:
                    state = torch.load(training_state_path, map_location="cpu")
                    error = (
                        "CUDA OOM" if "CUDA out of memory" in process.stderr else None
                    )
                    results.append(
                        {
                            "config": config_name,
                            "success": error is None,
                            "error": error,
                            "runtime_seconds": runtime,
                            "peak_mfu": state.get("peak_mfu", 0.0),
                            "peak_tflops": state.get("peak_tflops", 0.0),
                            "peak_memory_gb": state.get("peak_memory_gb", 0.0),
                            "total_tokens": state.get("total_tokens", 0),
                            "final_step": state.get("step", 0),
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "config": config_name,
                            "success": False,
                            "error": f"Failed to load metrics: {e}",
                            "runtime_seconds": runtime,
                        }
                    )
            else:
                error = (
                    "CUDA OOM"
                    if "CUDA out of memory" in process.stderr
                    else f"Exit code {process.returncode}"
                )
                results.append(
                    {
                        "config": config_name,
                        "success": False,
                        "error": error,
                        "runtime_seconds": runtime,
                    }
                )

        except subprocess.TimeoutExpired:
            results.append(
                {
                    "config": config_name,
                    "success": False,
                    "error": "Timeout",
                    "runtime_seconds": 1800,
                }
            )
        except Exception as e:
            results.append(
                {
                    "config": config_name,
                    "success": False,
                    "error": str(e),
                    "runtime_seconds": time.perf_counter() - run_start,
                }
            )

        # Clean up memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Save and display results
    results_file = (
        results_dir
        / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "runtime": time.perf_counter() - start_time,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 80}\nBenchmark Summary\n{'=' * 80}")
    print(
        f"{'Config':<30} {'Status':<15} {'MFU %':<8} {'TFLOPS':<10} {'Mem(GB)':<10} {'Runtime(s)':<12}"
    )
    print("-" * 80)
    for r in results:
        status = "Success!" if r["success"] else f"{r['error']}"
        mfu = f"{r.get('peak_mfu', 0):.1f}" if r["success"] else "-"
        tflops = f"{r.get('peak_tflops', 0):.1f}" if r["success"] else "-"
        memory = f"{r.get('peak_memory_gb', 0):.1f}" if r["success"] else "-"
        runtime = f"{r.get('runtime_seconds', 0):.1f}"
        print(
            f"{r['config']:<30} {status:<15} {mfu:<8} {tflops:<10} {memory:<10} {runtime:<12}"
        )
    print(
        f"{'=' * 80}\nResults saved to: {results_file}\nTotal runtime: {time.perf_counter() - start_time:.1f}s\n{'=' * 80}\n"
    )
