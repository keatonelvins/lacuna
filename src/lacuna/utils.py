import torch
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger
from collections import defaultdict, deque
from rich.pretty import Pretty
from rich.console import Console

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
        json.dump(state.model_dump(), f, indent=4)


def save_settings_json(path: Path, config: LacunaConfig) -> None:
    if not is_master():
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "settings.json").open("w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=4)


def load_state_json(path: Path) -> StateTracker:
    ts_path = path / "state.json"
    if ts_path.exists():
        with ts_path.open("r") as f:
            return StateTracker(**json.load(f))
    return StateTracker()


# Fast Best Fit Decreasing sample packing strategy
# From: https://github.com/huggingface/trl/blob/d15049bf71e6e33b2e6c10ff25a26d488bce8173/trl/data_utils.py#L450-L698
class _SegmentTree:
    def __init__(self, maxval: int):
        self.maxval = maxval
        self.tree_size = 1 << (maxval - 1).bit_length()
        self.tree = [0] * (2 * self.tree_size)

    def add(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = val
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            self.tree[i] = left if left >= right else right

    def remove(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = 0
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
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


def pad(tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    """Pad tensors to same length."""
    max_len = max(len(t) for t in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - len(tensor)
        padded.append(torch.cat([tensor, torch.full((pad_len,), padding_value, dtype=tensor.dtype)]))
    return torch.stack(padded)


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

    assert all(column.num_chunks == 1 for column in examples.columns)  # `pc.take` returns a ChunkedArray with a single chunk

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


@dataclass
class DataCollator:
    """Data collator w/ multipacking (lean on FA for attention mask)"""

    pad_token_id: int
    packing: bool = True

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        labels = [torch.tensor(example["labels"]) for example in examples]

        if self.packing:
            # Concatenate all sequences into single tensor (no padding)
            output = {
                "input_ids": torch.cat(input_ids, dim=0).unsqueeze(0),
                "labels": torch.cat(labels, dim=0).unsqueeze(0),
            }

            if "seq_lengths" in examples[0]:
                # Use packed sequence lengths for position IDs
                position_ids = self._get_position_ids_from_packed_seq_lengths([example["seq_lengths"] for example in examples])
                output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
            else:
                # Generate position IDs for individual sequences
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
                output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
        else:
            # Standard right padding (no position_ids needed for non-packed)
            output = {
                "input_ids": pad(input_ids, self.pad_token_id),
                "attention_mask": pad([torch.ones_like(ids) for ids in input_ids], 0),
                "labels": pad(labels, -100),
            }

        return output

    @staticmethod
    def _get_position_ids_from_packed_seq_lengths(
        batch_seq_lengths: list[list[int]],
    ) -> list[torch.Tensor]:
        """Generate position IDs for packed sequences."""
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        batch_seq_tensor = torch.tensor([seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths])
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_tensor.dtype)
        position_ids[0] = 0
        position_ids[batch_seq_tensor[:-1].cumsum(0)] = -(batch_seq_tensor[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        return list(position_ids.split(example_lengths))
