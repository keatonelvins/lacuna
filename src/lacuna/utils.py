import os
import json
from typing import Any
from pathlib import Path
from loguru import logger
from rich.pretty import Pretty
from rich.console import Console
from dataclasses import asdict
from collections import defaultdict, deque

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from transformers.utils.logging import disable_progress_bar

from .distributed import is_master
from .config import LacunaConfig
from .metrics import StateTracker


def setup_logger() -> None:
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        filter=lambda r: is_master(),
    )


def setup_env() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    disable_progress_bar()


def display_config(config: LacunaConfig) -> None:
    console = Console()
    with console.capture() as capture:
        console.print(Pretty(config, expand_all=True))  # omg Will you've outdone yourself
    logger.info("Starting training with config:\n" + capture.get().strip())


def save_state_json(path: Path, state: StateTracker) -> None:
    if not is_master():
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "state.json").open("w") as f:
        f.write(json.dumps(asdict(state), indent=4))


def save_settings_json(path: Path, config: LacunaConfig) -> None:
    if not is_master():
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "settings.json").open("w") as f:
        f.write(config.model_dump_json(indent=4))


def load_state_json(path: Path) -> StateTracker:
    ts_path = path / "state.json"
    if ts_path.exists():
        with ts_path.open("r") as f:
            return StateTracker(**json.load(f))
    return StateTracker()


def log_training_metrics(
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    metrics: dict[str, float],
) -> None:
    """Log training metrics in a colorful format."""
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


# ref: https://github.com/huggingface/trl/blob/main/trl/data_utils.py
class _SegmentTree:
    """
    A segment tree data structure that, when initialized as `_SegmentTree(maxval)`, efficiently finds the next larger
    value for a given input within the range [1, maxval].

    See [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) for more details.
    """

    def __init__(self, maxval: int):
        self.maxval = maxval
        # For non-power-of-2 values, we need to round up to the next power of 2 for the tree size
        self.tree_size = 1 << (maxval - 1).bit_length()
        self.tree = [0] * (2 * self.tree_size)

    def add(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = val
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            # Compare the values using if-else otherwise repeated calls to `builtins.max` become the bottleneck
            self.tree[i] = left if left >= right else right

    def remove(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = 0
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            # Compare the values using if-else otherwise repeated calls to `builtins.max` become the bottleneck
            self.tree[i] = left if left >= right else right

    def search(self, val):
        assert 0 < val <= self.maxval
        i = 1
        while i < self.tree_size:
            if self.tree[i << 1] >= val:
                i = i << 1
            else:
                i = (i << 1) + 1
        return self.tree[i]


def pack_bfd(examples: pa.Table, seq_length: int) -> pa.Table:
    """Pack sequences in a pyarrow Table using Best Fit Decreasing strategy."""
    columns = []
    list_column_idx = None
    for idx, column in enumerate(examples.columns):
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            column = pc.list_slice(column, 0, seq_length)
            if list_column_idx is None:
                list_column_idx = idx
        columns.append(column)
    examples = pa.Table.from_arrays(columns, names=examples.column_names)

    ids = np.arange(len(examples))
    assert list_column_idx is not None
    lengths = pc.list_value_length(examples[list_column_idx]).combine_chunks()
    examples = examples.append_column("seq_lengths", lengths)  # Allows us to later construct `position_ids`
    lengths = pc.make_struct(lengths, ids)
    lengths = lengths.sort("descending", by=0)

    segment_tree = _SegmentTree(seq_length)
    segment_tree.add(seq_length)  # the max, `seq_length` bin is always available
    space_to_bin = defaultdict(deque)

    # Bin is represented as a dict (of example ids and sum of their lengths) to allow in-place updates
    bins: list[dict] = []
    for length, idx in zip(lengths.field(0).to_numpy(), lengths.field(1).to_numpy()):
        space = segment_tree.search(length)

        if space < seq_length:
            # Use existing bin with exactly this amount of space
            bin = space_to_bin[space].popleft()
        else:
            # Create a new bin
            bin = {"ids": [], "length": 0}
            bins.append(bin)

        bin["ids"].append(idx)
        bin["length"] += length
        if space < seq_length and not space_to_bin[space]:
            segment_tree.remove(space)

        space = space - length
        space_to_bin[space].append(bin)
        if space > 0:
            segment_tree.add(space)

    examples = pc.take(examples, [id_ for bin in bins for id_ in bin["ids"]])
    offsets = np.array([0] + [bin["length"] for bin in bins])
    offsets = np.cumsum(offsets)

    assert all(
        column.num_chunks == 1 for column in examples.columns
    )  # `pc.take` returns a ChunkedArray with a single chunk

    lengths = examples["seq_lengths"].chunks[0]
    examples = examples.drop_columns("seq_lengths")
    lengths = pa.ListArray.from_arrays(np.cumsum([0] + [len(bin["ids"]) for bin in bins], dtype=np.int32), lengths)

    columns = []
    for column in examples.columns:
        column = column.chunks[0]
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            dtype = column.offsets.type.to_pandas_dtype()
            column = type(column).from_arrays(offsets.astype(dtype), column.values)
        columns.append(column)
    return pa.Table.from_arrays(columns + [lengths], names=examples.column_names + ["seq_lengths"])
