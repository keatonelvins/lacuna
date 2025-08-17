"""Data loading, tokenization, and packing for training."""

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import PretrainConfig, SFTConfig


def setup_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Setup tokenizer with proper pad token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
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
                        "input_ids": torch.tensor(input_chunk, dtype=torch.long),
                        "labels": torch.tensor(label_chunk, dtype=torch.long),
                    }
                )

        logger.info(f"Created {len(self.chunks)} chunks of length {seq_len}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.chunks[idx]


def setup_dataloader(
    config: PretrainConfig | SFTConfig, micro_batch_size: int
) -> DataLoader:
    """Setup simple dataloader for training."""

    # Setup tokenizer
    tokenizer = setup_tokenizer(config.model.name)

    # Create dataset based on config type
    if isinstance(config, PretrainConfig):
        dataset = PretrainDataset(
            config.data.dataset_name, config.data.split, tokenizer, config.data.seq_len
        )
    else:  # SFTConfig - TODO: implement later
        raise NotImplementedError("SFT dataset not implemented yet")

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Dataloader created with {len(dataset)} samples, micro_batch_size={micro_batch_size}"
    )

    return dataloader
