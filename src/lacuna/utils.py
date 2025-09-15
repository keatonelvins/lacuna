import json
import os
from pathlib import Path
from datetime import datetime
from loguru import logger
from rich.pretty import Pretty
from rich.console import Console
from collections import defaultdict, deque

import torch
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .distributed import is_master
from .config import LacunaConfig


def master_only(fn):
    """Decorator to run a function only on the master process."""

    def wrapper(*args, **kwargs):
        if not is_master():
            return
        return fn(*args, **kwargs)

    return wrapper


def get_run_dir() -> Path:
    """Create and return a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(".lacuna_cache/runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    active_link = Path(".lacuna_cache/active_run")
    if active_link.exists() or active_link.is_symlink():
        active_link.unlink()
    active_link.symlink_to(run_dir.relative_to(active_link.parent))

    return run_dir


def setup_logger(run_dir: Path = None) -> None:
    """Setup logging to console and run directory."""
    logger.remove()  # Remove default handler

    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        filter=lambda r: is_master(),
    )

    if run_dir:
        logger.add(
            run_dir / "run.log",
            format="{time:HH:mm:ss} | {level} | {message}",
            level="INFO",
            filter=lambda r: is_master(),
            rotation="100 MB",
        )


def setup_env(config: LacunaConfig) -> Path:
    # high -> TF32, highest -> FP32
    torch.set_float32_matmul_precision("high")

    run_dir = get_run_dir()
    setup_logger(run_dir)
    save_settings_json(run_dir, config)

    config.checkpoint.prepare_save_dir()  # clear save_dir if not resuming
    os.makedirs(".lacuna_cache", exist_ok=True)

    return run_dir


def display_config(config: LacunaConfig) -> None:
    console = Console(force_terminal=False, no_color=True)
    with console.capture() as capture:
        console.print(Pretty(config, expand_all=True))  # omg Will you've outdone yourself
    logger.info("Starting training with config:\n" + capture.get().strip())


@master_only
def save_metrics_jsonl(run_dir: Path, step: int, loss: float, grad_norm: float, lr: float, metrics: dict) -> None:
    metrics_data = {
        "step": step,
        "loss": loss,
        "grad_norm": float(grad_norm),
        "lr": lr,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    metrics_file = run_dir / "metrics.jsonl"
    with metrics_file.open("a") as f:
        f.write(json.dumps(metrics_data) + "\n")


@master_only
def save_settings_json(path: Path, config: LacunaConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "settings.json").open("w") as f:
        f.write(config.model_dump_json(indent=4))


def log_training_metrics(
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    metrics: dict[str, float],
    run_dir: Path = None,
) -> None:
    log_parts = [
        f"Step {step:>6}",
        f"Loss: {loss:7.4f}",
        f"Grad: {grad_norm:8.4f}",
        f"LR: {lr:9.2e}",
        f"Mem: {metrics.get('max_reserved_gb', 0.0):5.1f}GB ({metrics.get('max_reserved_pct', 0.0):3.0f}%)",
        f"MFU: {metrics.get('mfu_pct', 0.0):5.1f}%",
        f"Data: {metrics.get('data_pct', 0.0):5.1f}%",
    ]
    logger.info(" | ".join(log_parts))

    if run_dir:
        save_metrics_jsonl(run_dir, step, loss, grad_norm, lr, metrics)


# some gpt-5 code for bfd packing
class IntSucc:
    __slots__ = ("N", "bits")

    def __init__(self, maxval: int):
        assert maxval >= 1
        self.N, self.bits = maxval, 0

    def add(self, i: int):
        self.bits |= 1 << (i - 1)

    def discard(self, i: int):
        self.bits &= ~(1 << (i - 1))

    def next_geq(self, x: int) -> int:
        y = self.bits >> (x - 1)
        assert y, "no successor present (missing sentinel?)"
        return x + ((y & -y).bit_length() - 1)


def _take(arr, idx):
    out = pc.take(arr, pa.array(idx, type=pa.int64()))
    return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out


def pack_bfd(examples: pa.Table, seq_len: int) -> pa.Table:
    ids = pc.list_slice(examples["input_ids"], 0, seq_len)
    has_masks = "assistant_masks" in examples.column_names
    masks = pc.list_slice(examples["assistant_masks"], 0, seq_len) if has_masks else None

    lens = pc.list_value_length(ids).to_numpy()
    order = np.argsort(-lens)

    succ = IntSucc(seq_len)
    succ.add(seq_len)  # sentinel enables new bins
    by_space = defaultdict(deque)  # space -> deque[bins]
    bins = []  # each: {"ids": [...], "len": int}

    for i in order:
        L = int(lens[i])
        if not L:
            continue
        s = succ.next_geq(L)
        b = by_space[s].popleft() if s < seq_len else {"ids": [], "len": 0}
        if s < seq_len and not by_space[s]:
            succ.discard(s)
        b["ids"].append(int(i))
        b["len"] += L
        if s == seq_len:
            bins.append(b)
        ns = s - L
        by_space[ns].append(b)
        if ns:
            succ.add(ns)

    reorder = [j for b in bins for j in b["ids"]]
    ids_taken = _take(ids, reorder)
    if has_masks:
        masks_taken = _take(masks, reorder)

    # offsets (match ListArray vs LargeListArray via dtype)
    tok_counts = [b["len"] for b in bins]
    odtype = ids_taken.offsets.type.to_pandas_dtype()
    offs = np.cumsum([0] + tok_counts, dtype=odtype)

    LA = type(ids_taken)
    packed_ids = LA.from_arrays(offs, ids_taken.values)

    # position_ids: reset to 0 at each original example boundary
    dl = lens[reorder]
    T = int(offs[-1])
    pos = np.ones(T, dtype=np.int64)
    pos[0] = 0
    if dl.size > 1:
        cut = dl[:-1].cumsum()
        pos[cut] = -(dl[:-1] - 1)
    pos = pos.cumsum()
    position_ids = LA.from_arrays(offs, pa.array(pos, type=pa.int64()))

    if has_masks:
        packed_masks = LA.from_arrays(offs, masks_taken.values)
        return pa.Table.from_arrays(
            [packed_ids, position_ids, packed_masks], names=["input_ids", "position_ids", "assistant_masks"]
        )
    return pa.Table.from_arrays([packed_ids, position_ids], names=["input_ids", "position_ids"])
