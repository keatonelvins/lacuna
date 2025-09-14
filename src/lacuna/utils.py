import json
from pathlib import Path
from loguru import logger
from rich.pretty import Pretty
from rich.console import Console
from dataclasses import asdict
from collections import defaultdict, deque

import torch
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .distributed import is_master
from .config import LacunaConfig
from .metrics import StateTracker


def master_only(fn):
    """Decorator to run a function only on the master process."""

    def wrapper(*args, **kwargs):
        if not is_master():
            return
        return fn(*args, **kwargs)

    return wrapper


def setup_logger() -> None:
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        filter=lambda r: is_master(),
    )
    # Also log to file for debugging
    logger.add(
        "runs.log", format="{time:HH:mm:ss} | {level} | {message}", level="INFO", filter=lambda r: is_master(), rotation="1 MB"
    )


def setup_env(config: LacunaConfig) -> None:
    # high -> TF32, highest -> FP32
    torch.set_float32_matmul_precision("high")

    setup_logger()
    config.checkpoint.prepare_save_dir()  # clear save_dir if not resuming


def display_config(config: LacunaConfig) -> None:
    console = Console()
    with console.capture() as capture:
        console.print(Pretty(config, expand_all=True))  # omg Will you've outdone yourself
    logger.info("Starting training with config:\n" + capture.get().strip())


@master_only
def save_state_json(path: Path, state: StateTracker) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "state.json").open("w") as f:
        f.write(json.dumps(asdict(state), indent=4))


@master_only
def save_settings_json(path: Path, config: LacunaConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "settings.json").open("w") as f:
        f.write(config.model_dump_json(indent=4))


def load_state_json(path: Path) -> StateTracker:
    tracker_path = path / "state.json"
    if tracker_path.exists():
        with tracker_path.open("r") as f:
            return StateTracker(**json.load(f))
    return StateTracker()


def log_training_metrics(
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    metrics: dict[str, float],
) -> None:
    log_parts = [
        f"\033[91mStep {step:>6}\033[0m",
        f"\033[92mLoss: {loss:7.4f}\033[0m",
        f"\033[93mGrad: {grad_norm:8.4f}\033[0m",
        f"\033[94mLR: {lr:9.2e}\033[0m",
        f"\033[36mMem: {metrics.get('max_reserved_gb', 0.0):5.1f}GB ({metrics.get('max_reserved_pct', 0.0):3.0f}%)\033[0m",
        f"\033[92mMFU: {metrics.get('mfu_pct', 0.0):5.1f}%\033[0m",
        f"\033[33mData: {metrics.get('data_pct', 0.0):5.1f}%\033[0m",
    ]
    logger.info(" | ".join(log_parts))


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
