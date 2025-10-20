"""Misc utils for trainer and data."""

import os
import json
import time
import torch
import random
import tomli_w
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch.distributed as dist
from contextlib import contextmanager
from torchtitan.components.metrics import build_device_memory_monitor

from lacuna.config import TrainConfig


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_master() -> bool:
    return get_rank() == 0


def master_only(fn):
    """Decorator to run a function only on the master process."""

    def wrapper(*args, **kwargs):
        if not is_master():
            return
        return fn(*args, **kwargs)

    return wrapper


@contextmanager
def run_master_first():
    """Context manager to run the master process first, then the rest."""
    if dist.is_initialized():
        if is_master():
            yield
        dist.barrier()
        if not is_master():
            yield
        dist.barrier()
    else:
        yield


def setup_logger(run_dir: Path) -> None:
    """Setup logging to console and run directory."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:HH:mm:ss} | {level} | {message}",
        filter=lambda r: is_master(),
    )
    if run_dir is not None:
        logger.add(
            run_dir / "run.log",
            format="{time:HH:mm:ss} | {level} | {message}",
            filter=lambda r: is_master(),
        )


@master_only
def setup_run_dir(config: TrainConfig, run_dir: Path) -> Path:
    """Create and return a timestamped run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.toml", "wb") as f:
        tomli_w.dump(config.model_dump(exclude_defaults=True, mode="json"), f)


@master_only
def log_train_metrics(step: int, metrics: dict, run_dir: Path) -> None:
    log_parts = [
        f"Step {step:>6}",
        f"Loss: {metrics['train/loss']:7.4f}",
        f"Grad: {metrics['train/grad_norm']:8.4f}",
        f"Tok/s: {metrics['perf/throughput(tps)']:6.2f}",
        f"Mem: {metrics['memory/max_active(GiB)']:6.2f}GiB",
        f"Toks: {metrics['train/ntokens_micro_batch']:4d}",
    ]
    logger.info(" | ".join(log_parts))
    append_jsonl(run_dir, metrics, "metrics")


@master_only
def log_eval_metrics(step: int, metrics: dict, run_dir: Path) -> None:
    log_parts = [
        f"Step {step:>6}",
        f"Eval Loss: {metrics['eval/loss']:7.4f}",
        f"Perplexity: {metrics['eval/perplexity']:9.3f}",
        f"Token Acc: {metrics['eval/token_accuracy'] * 100:6.2f}%",
    ]
    logger.info(" | ".join(log_parts))
    append_jsonl(run_dir, {"step": step, **metrics}, "eval")


@master_only
def append_jsonl(run_dir: Path, metrics: dict, name: str = "metrics") -> None:
    metrics_data = {"timestamp": datetime.now().isoformat(), **metrics}
    metrics_file = run_dir / f"{name}.jsonl"
    with metrics_file.open("a") as f:
        f.write(json.dumps(metrics_data) + "\n")


def init_dist() -> None:
    """Initialize distributed process group."""
    # high -> TF32, highest -> FP32
    torch.backends.cuda.matmul.fp32_precision = "tf32"

    if "LOCAL_RANK" not in os.environ:
        return  # single GPU

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", device_id=local_rank)
    logger.info(f"Initialized distributed training: rank={get_rank()}, world_size={get_world_size()}")


def destroy_dist() -> None:
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> int:
    """Set seeds for reproducibility across all RNGs."""
    if dist.is_initialized():
        seed_tensor = torch.tensor(seed, dtype=torch.long, device="cuda")
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


# ref: https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/metrics.py
class MetricsProcessor:
    """Metrics processor/tracker for training."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.data_loading_times = []
        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor = build_device_memory_monitor()
        self.device_memory_monitor.reset_peak_stats()

    def get_metrics(self) -> dict:
        time_delta = time.perf_counter() - self.time_last_log
        tps = self.ntokens_since_last_log / time_delta
        time_end_to_end = time_delta / self.config.metrics.log_every
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta
        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "perf/throughput(tps)": tps,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading_pct(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active_pct(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved_pct(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        return metrics


# ref: https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/trainer/model.py
def get_moe_stats(model: torch.nn.Module) -> dict:
    """Collect MoE load balance metrics from torchtitan MoE layers."""
    per_layer_max_vio = []
    for layer in model.model.layers:
        if not hasattr(layer.mlp, "tokens_per_expert"):
            continue
        tokens_per_expert = layer.mlp.tokens_per_expert

        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.item())
        tokens_per_expert.zero_()

    if not per_layer_max_vio:
        return {}

    return {"moe/load_balance_max_vio": torch.tensor(per_layer_max_vio).mean().item()}
