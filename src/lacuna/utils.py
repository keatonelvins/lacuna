import os
import json
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


def pack_bfd(examples: pa.Table, seq_len: int) -> pa.Table:
    """
    Pack `input_ids` examples into fixed-length bins (Best-Fit Decreasing) and
    return only two columns: `input_ids` (packed) and `position_ids`.

    - Truncates each example to `seq_len`.
    - Concatenates multiple examples into each packed row up to `seq_len`.
    - `position_ids` reset to 0 at each original example boundary.
    """
    # Hardcoded to our single use case: a table with list column `input_ids`.
    input_ids_col = pc.list_slice(examples["input_ids"], 0, seq_len)

    # Compute lengths and sort ids by decreasing length
    lengths_np = pc.list_value_length(input_ids_col).to_numpy()
    ids_np = np.arange(len(lengths_np))
    order_desc = np.argsort(-lengths_np)

    # Best-Fit Decreasing using a segment tree for efficiency
    segment_tree = _SegmentTree(seq_len)
    segment_tree.add(seq_len)  # the max-capacity bin is always available
    space_to_bin = defaultdict(deque)
    bins: list[dict] = []  # each bin: {"ids": list[int], "length": int}

    for idx in order_desc:
        length = int(lengths_np[idx])
        if length == 0:
            continue

        space = segment_tree.search(length)
        if space < seq_len:
            binref = space_to_bin[space].popleft()
        else:
            binref = {"ids": [], "length": 0}
            bins.append(binref)

        binref["ids"].append(int(ids_np[idx]))
        binref["length"] += length

        if space < seq_len and not space_to_bin[space]:
            segment_tree.remove(space)

        new_space = space - length
        space_to_bin[new_space].append(binref)
        if new_space > 0:
            segment_tree.add(new_space)

    # Reorder examples by packed-bin grouping, then rebuild a single list per bin
    reorder = [eid for b in bins for eid in b["ids"]]
    taken = pc.take(input_ids_col, pa.array(reorder, type=pa.int64()))
    # Ensure we have a single contiguous array regardless of chunking
    if isinstance(taken, pa.ChunkedArray):
        list_arr = taken.combine_chunks()
    else:
        list_arr = taken

    # Build packed offsets: one list per bin
    bin_token_counts = [b["length"] for b in bins]
    offsets_dtype = list_arr.offsets.type.to_pandas_dtype()
    packed_offsets = np.cumsum([0] + bin_token_counts, dtype=offsets_dtype)

    list_array_cls = type(list_arr)
    packed_input_ids = list_array_cls.from_arrays(packed_offsets, list_arr.values)

    # Build position_ids: 0..len-1 for each original example, grouped per packed row
    doc_lengths_ordered = lengths_np[reorder]
    total_tokens = int(packed_offsets[-1]) if len(packed_offsets) > 0 else 0
    if total_tokens > 0:
        pos_vals = np.ones(total_tokens, dtype=np.int64)
        pos_vals[0] = 0
        if len(doc_lengths_ordered) > 1:
            ends = doc_lengths_ordered[:-1].cumsum()
            pos_vals[ends] = -(doc_lengths_ordered[:-1] - 1)
        pos_vals = pos_vals.cumsum()
    else:
        pos_vals = np.array([], dtype=np.int64)

    position_ids = list_array_cls.from_arrays(packed_offsets, pa.array(pos_vals, type=pa.int64()))

    return pa.Table.from_arrays([packed_input_ids, position_ids], names=["input_ids", "position_ids"])
