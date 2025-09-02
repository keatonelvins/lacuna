import torch
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger
from collections import defaultdict, deque
from rich.pretty import Pretty
from rich.console import Console

from .distributed import get_rank
from .config import SFTConfig, PretrainConfig


def _rank_filter(_):
    return get_rank() == 0


def setup_logger() -> None:
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        filter=_rank_filter,
    )


def display_config(config: SFTConfig | PretrainConfig) -> None:
    console = Console()
    with console.capture() as capture:
        console.print(
            Pretty(config, expand_all=True)
        )  # omg Will you've outdone yourself
    logger.info("Starting training with config:\n" + capture.get().strip())


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
        padded.append(
            torch.cat(
                [tensor, torch.full((pad_len,), padding_value, dtype=tensor.dtype)]
            )
        )
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
    examples = examples.append_column(
        "seq_lengths", lengths
    )  # Allows us to later construct `position_ids`
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
    lengths = pa.ListArray.from_arrays(
        np.cumsum([0] + [len(bin["ids"]) for bin in bins], dtype=np.int32), lengths
    )

    columns = []
    for column in examples.columns:
        column = column.chunks[0]
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            dtype = column.offsets.type.to_pandas_dtype()
            column = type(column).from_arrays(offsets.astype(dtype), column.values)
        columns.append(column)
    return pa.Table.from_arrays(
        columns + [lengths], names=examples.column_names + ["seq_lengths"]
    )
