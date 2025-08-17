"""Data loading, tokenization, and packing for training."""

import logging
from typing import Iterator

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import PretrainConfig, SFTConfig

logger = logging.getLogger("lacuna")


def setup_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Setup tokenizer with proper pad token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_pretrain_batch(
    examples: dict, tokenizer: PreTrainedTokenizerBase, seq_len: int
) -> dict:
    """Tokenize batch for pretraining (simple concatenation and chunking)."""
    # Concatenate all text
    texts = examples["text"]
    combined_text = "\n".join(texts)

    # Tokenize and chunk into fixed sequences
    tokens = tokenizer(
        combined_text, truncation=False, padding=False, return_tensors="np"
    )["input_ids"][0]

    # Create fixed-length chunks
    chunks = []
    for i in range(0, len(tokens) - seq_len + 1, seq_len):
        chunk = tokens[i : i + seq_len]
        if len(chunk) == seq_len:
            chunks.append(chunk)

    return {"input_ids": chunks}


def tokenize_sft_batch(
    examples: dict, tokenizer: PreTrainedTokenizerBase, seq_len: int
) -> dict:
    """Tokenize batch for SFT (chat format)."""
    # For now, use simple text processing
    # TODO: Implement proper chat template parsing and loss masking
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_tensors="pt",
    )

    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["input_ids"].clone(),  # For now, no masking
    }


def setup_dataloader(config: PretrainConfig | SFTConfig) -> Iterator[dict]:
    """Setup dataloader for training."""
    logger.info(f"Loading dataset: {config.data.dataset_name}")

    dataset = load_dataset(
        config.data.dataset_name,
        split=config.data.split,
        streaming=True,
    )

    # Setup tokenizer
    tokenizer = setup_tokenizer(config.model.name)

    # Tokenize dataset
    if isinstance(config, PretrainConfig):

        def tokenize_fn(examples: dict) -> dict:
            return tokenize_pretrain_batch(examples, tokenizer, config.data.seq_len)
    else:  # SFTConfig

        def tokenize_fn(examples: dict) -> dict:
            return tokenize_sft_batch(examples, tokenizer, config.data.seq_len)

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.data.micro_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Dataloader created with micro_batch_size={config.data.micro_batch_size}"
    )

    return iter(dataloader)
