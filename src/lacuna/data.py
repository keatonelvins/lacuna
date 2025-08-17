"""Data loading, tokenization, and packing for training."""

import torch
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from collections import defaultdict, deque
from dataclasses import dataclass
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Any

from .config import PretrainConfig, SFTConfig


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


def _pad(
    tensors: list[torch.Tensor], padding_value: int, padding_side: str = "right"
) -> torch.Tensor:
    """Pad tensors to same length."""
    max_len = max(len(t) for t in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - len(tensor)
        if padding_side == "right":
            padded.append(
                torch.cat(
                    [tensor, torch.full((pad_len,), padding_value, dtype=tensor.dtype)]
                )
            )
        else:
            padded.append(
                torch.cat(
                    [torch.full((pad_len,), padding_value, dtype=tensor.dtype), tensor]
                )
            )
    return torch.stack(padded)


def _pack_bfd(examples: pa.Table, seq_length: int) -> pa.Table:
    """Pack sequences in a pyarrow Table using Best Fit Decreasing strategy."""
    columns = []
    list_column_idx = None
    for idx, column in enumerate(examples.columns):
        if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(
            column.type
        ):
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


@dataclass
class DataCollator:
    """Data collator (packing depends on Flash Attention)"""

    pad_token_id: int
    packing: bool = True
    return_position_ids: bool = True

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        labels = [torch.tensor(example["labels"]) for example in examples]

        if self.packing:
            # Concatenate all sequences into single tensor (no padding)
            output = {
                "input_ids": torch.cat(input_ids, dim=0).unsqueeze(0),
                "labels": torch.cat(labels, dim=0).unsqueeze(0),
            }

            if self.return_position_ids:
                if "seq_lengths" in examples[0]:
                    # Use packed sequence lengths for position IDs
                    position_ids = self._get_position_ids_from_packed_seq_lengths(
                        [example["seq_lengths"] for example in examples]
                    )
                    output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
                else:
                    # Generate position IDs for individual sequences
                    position_ids = [torch.arange(len(ids)) for ids in input_ids]
                    output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
        else:
            # Traditional padding
            output = {
                "input_ids": _pad(input_ids, self.pad_token_id),
                "attention_mask": _pad([torch.ones_like(ids) for ids in input_ids], 0),
                "labels": _pad(labels, -100),
            }

            if self.return_position_ids:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
                output["position_ids"] = _pad(position_ids, 0)

        return output

    @staticmethod
    def _get_position_ids_from_packed_seq_lengths(
        batch_seq_lengths: list[list[int]],
    ) -> list[torch.Tensor]:
        """Generate position IDs for packed sequences."""
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        batch_seq_lengths = torch.tensor(
            [
                seq_length
                for seq_lengths in batch_seq_lengths
                for seq_length in seq_lengths
            ]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        return list(position_ids.split(example_lengths))


def setup_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Setup tokenizer with proper pad token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        logger.info(f"No pad token found, using eos token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class PretrainDataset(Dataset):
    """Simple dataset loader for continued pretraining."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
    ):
        logger.info(f"Loading dataset: {dataset_name}")

        # TODO: support streaming
        dataset = load_dataset(dataset_name, split=split)

        logger.info("Tokenizing dataset...")
        all_tokens = []

        for text in dataset["text"]:
            text_with_eos = text + tokenizer.eos_token
            tokens = tokenizer(text_with_eos, truncation=False, padding=False)[
                "input_ids"
            ]
            all_tokens.extend(tokens)

        tokens = all_tokens

        # Chunk into fixed-length sequences
        self.chunks = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            input_chunk = tokens[i : i + seq_len]
            label_chunk = tokens[
                i + 1 : i + seq_len + 1
            ]  # Labels are input shifted by 1

            if len(input_chunk) == seq_len and len(label_chunk) == seq_len:
                self.chunks.append(
                    {
                        "input_ids": input_chunk,
                        "labels": label_chunk,
                    }
                )

        logger.info(f"Created {len(self.chunks)} chunks of length {seq_len}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, list]:
        return self.chunks[idx]


class SFTDataset(Dataset):
    """SFT dataset w/ packing support

    TODO: assistant-only loss
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        packing: bool = True,
    ):
        logger.info(f"Loading SFT dataset: {dataset_name}")

        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        if "messages" not in dataset.column_names:
            logger.error(f"Expected 'messages' column, found: {dataset.column_names}")
            raise ValueError("SFTDataset requires a 'messages' column in OpenAI format")

        # Tokenize conversations
        logger.info("Tokenizing conversations...")
        tokenized_samples = []

        for example in dataset:
            try:
                input_ids = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=True,
                    padding=False,
                    truncation=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                continue

            # TODO: support assistant-only loss
            labels = input_ids.copy()

            sample = {
                "input_ids": input_ids,
                "labels": labels,
            }

            # Filter out samples that are too long
            if len(sample["input_ids"]) <= seq_len:
                tokenized_samples.append(sample)
            else:
                logger.warning(
                    f"Sample is too long, skipping: {len(sample['input_ids'])} > {seq_len}"
                )

        # Apply packing or padding
        if packing:
            logger.info("Packing samples...")
            pre_packing_sample_count = len(tokenized_samples)
            self.samples = self._pack_samples(tokenized_samples, seq_len)
            post_packing_sample_count = len(self.samples)
            logger.info(
                f"Packed {pre_packing_sample_count} samples into {post_packing_sample_count} samples"
            )
        else:
            logger.info("Padding samples...")
            self.samples = tokenized_samples

        logger.info(f"Created {len(self.samples)} samples")

    def _pack_samples(self, samples: list[dict], seq_len: int) -> list[dict]:
        """Pack samples into sequences of length `seq_len`"""
        input_ids_list = [sample["input_ids"] for sample in samples]
        labels_list = [sample["labels"] for sample in samples]
        table_data = {
            "input_ids": input_ids_list,
            "labels": labels_list,
        }

        table = pa.table(table_data)

        packed_table = _pack_bfd(table, seq_len)

        packed_samples = []
        input_ids_arrays = packed_table["input_ids"].to_pylist()
        labels_arrays = packed_table["labels"].to_pylist()
        seq_lengths_arrays = packed_table["seq_lengths"].to_pylist()

        for input_ids, labels, seq_lengths in zip(
            input_ids_arrays, labels_arrays, seq_lengths_arrays
        ):
            packed_samples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "seq_lengths": seq_lengths,
                }
            )

        return packed_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def setup_dataloader(
    config: PretrainConfig | SFTConfig, micro_batch_size: int
) -> DataLoader:
    tokenizer = setup_tokenizer(config.model.name)

    if isinstance(config, PretrainConfig):
        dataset = PretrainDataset(
            config.data.dataset_name, config.data.split, tokenizer, config.data.seq_len
        )
        packing = False  # Pretrain just concatenates and chunks
    else:  # SFTConfig
        dataset = SFTDataset(
            config.data.dataset_name,
            config.data.split,
            tokenizer,
            config.data.seq_len,
            packing=config.data.packing,
        )
        packing = config.data.packing

    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        packing=packing,
        return_position_ids=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    logger.info(
        f"Dataloader created with {len(dataset)} samples, micro_batch_size={micro_batch_size}, packing={packing}"
    )

    return dataloader
