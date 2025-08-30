"""Data loading, tokenization, and packing for training."""

import torch
import pyarrow as pa
import pyarrow.types
from dataclasses import dataclass
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Any

from .config import PretrainConfig, SFTConfig
from .distributed import get_world_size
from .utils import pad, pack_bfd


class RandomDataset(Dataset):
    """Dataset that generates random tokens for testing and benchmarking."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, list]:
        torch.manual_seed(idx)  # for reproducibility

        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,)).tolist()

        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


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
                position_ids = self._get_position_ids_from_packed_seq_lengths(
                    [example["seq_lengths"] for example in examples]
                )
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
        batch_seq_tensor = torch.tensor(
            [
                seq_length
                for seq_lengths in batch_seq_lengths
                for seq_length in seq_lengths
            ]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_tensor.dtype)
        position_ids[0] = 0
        position_ids[batch_seq_tensor[:-1].cumsum(0)] = -(batch_seq_tensor[:-1] - 1)
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

            if len(input_chunk) == seq_len == len(label_chunk):
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
    """SFT dataset w/ packing support and assistant-only loss masking."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        packing: bool = True,
    ):
        logger.info(f"Loading SFT dataset: {dataset_name}")

        dataset = load_dataset(dataset_name, split=split)

        if "messages" not in dataset.column_names:
            logger.error(f"Expected 'messages' column, found: {dataset.column_names}")
            raise ValueError("SFTDataset requires a 'messages' column in OpenAI format")

        logger.info("Tokenizing with assistant-only loss...")
        tokenized_samples = []

        for example in dataset:
            try:
                formatted_text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )

                # Tokenize with offset mapping to track text positions
                tokenized = tokenizer(
                    formatted_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )

                input_ids = tokenized["input_ids"]
                offset_mapping = tokenized["offset_mapping"]

                # Create assistant-only loss masks
                labels = self._get_assistant_masks(
                    example["messages"],
                    formatted_text,
                    input_ids,
                    offset_mapping,
                    tokenizer,
                )

            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                continue

            sample = {
                "input_ids": input_ids,
                "labels": labels,
            }

            # Filter out over-long samples
            if len(sample["input_ids"]) <= seq_len:
                tokenized_samples.append(sample)
            else:
                logger.warning(
                    f"Sample is too long, skipping: {len(sample['input_ids'])} > {seq_len}"
                )

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
        """Pack samples into sequences of length `seq_len` using Best Fit Decreasing"""
        input_ids_list = [sample["input_ids"] for sample in samples]
        labels_list = [sample["labels"] for sample in samples]
        table_data = {
            "input_ids": input_ids_list,
            "labels": labels_list,
        }

        table = pa.table(table_data)

        packed_table = pack_bfd(table, seq_len)

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

    def _get_assistant_masks(
        self,
        messages: list[dict],
        formatted_text: str,
        input_ids: list[int],
        offset_mapping: list[tuple[int, int]],
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]:
        """Create labels with assistant-only loss masking."""
        # Start with all tokens masked
        labels = [-100] * len(input_ids)

        # Find assistant message boundaries in the formatted text
        assistant_ranges = []
        for message in messages:
            if message["role"] == "assistant":
                content = message["content"]
                # Find this assistant content in the formatted text
                start_pos = formatted_text.find(content)
                if start_pos != -1:
                    end_pos = start_pos + len(content)
                    assistant_ranges.append((start_pos, end_pos))

        # Map text positions to token positions and unmask assistant tokens
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            # Check if this token falls within any assistant message
            for assist_start, assist_end in assistant_ranges:
                if start_char >= assist_start and end_char <= assist_end:
                    labels[token_idx] = input_ids[token_idx]
                    break
                # Include EOS token immediately after assistant messages
                elif start_char >= assist_end and start_char < assist_end + 10:
                    # Check if this token is EOS
                    if input_ids[token_idx] == tokenizer.eos_token_id:
                        labels[token_idx] = input_ids[token_idx]
                        break

        return labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def setup_dataloader(
    config: PretrainConfig | SFTConfig, micro_batch_size: int
) -> tuple[DataLoader, PreTrainedTokenizerBase, DistributedSampler | None]:
    tokenizer = setup_tokenizer(config.model.name)

    # Check if we should use random data
    if config.data.use_random_data:
        logger.info("Using random data for testing/benchmarking")
        dataset = RandomDataset(
            vocab_size=tokenizer.vocab_size,
            seq_len=config.data.seq_len,
            num_samples=10000,
        )
        packing = False
    elif isinstance(config, PretrainConfig):
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

    collator = DataCollator(pad_token_id=tokenizer.pad_token_id, packing=packing)

    # Use distributed sampler for multi-GPU training
    sampler = None
    shuffle = True
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False  # Don't shuffle when using sampler

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        collate_fn=collator,
    )

    logger.info(
        f"Dataloader created with {len(dataset)} samples, batch_size={micro_batch_size}"
    )

    return dataloader, tokenizer, sampler
