"""Data loading, tokenization, and packing for training."""

from loguru import logger
from functools import partial

import torch
from datasets import load_dataset, interleave_datasets
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import PretrainConfig, SFTConfig
from .distributed import get_rank, get_world_size


def _get_iterable_dataset(config: PretrainConfig | SFTConfig) -> IterableDataset:
    """Get single IterableDataset from merged hf datasets."""
    loader = partial(load_dataset, split=config.data.split, streaming=config.data.stream)
    datasets = [loader(name) for name in config.data.dataset_names]

    if not config.data.stream:
        datasets = [dataset.to_iterable_dataset(num_shards=get_world_size()) for dataset in datasets]

    dataset = interleave_datasets(datasets)

    if dataset.num_shards != get_world_size():
        logger.warning(f"Dataset has {dataset.num_shards} shards, but world size is {get_world_size()}")

    return dataset


class PretrainDataset(IterableDataset, Stateful):
    """Stateful pretraining dataset w/ packing."""

    def __init__(
        self,
        config: PretrainConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.config = config
        self.tokenizer = tokenizer

        logger.info(f"Loading datasets: {config.data.dataset_names}")
        dataset = _get_iterable_dataset(config)
        self._data = split_dataset_by_node(dataset, get_rank(), get_world_size())
        self._data = self._data.map(self._encode, batched=True, batch_size=1000)
        self._data = self._data.with_format("torch")

        self.num_shards = self._data.num_shards
        self.seq_len = config.data.seq_len

        self._sample_idx = 0
        self._token_buffer = []

    def _encode(self, examples):
        return self.tokenizer(examples["text"], add_special_tokens=False, truncation=False, padding=False)

    def __iter__(self):
        for sample in self._data:
            sample_tokens = sample["input_ids"]
            self._token_buffer.extend(sample_tokens)
            self._token_buffer.append(self.tokenizer.eos_token_id)
            self._sample_idx += 1

            while len(self._token_buffer) > self.seq_len:
                seq_len_tokens = torch.LongTensor(self._token_buffer[: self.seq_len + 1])
                self._token_buffer = self._token_buffer[self.seq_len + 1 :]
                input_ids = seq_len_tokens[:-1]
                labels = seq_len_tokens[1:]
                yield {"input_ids": input_ids, "labels": labels}

    def state_dict(self):
        state_dict = {"token_buffer": self._token_buffer}
        state_dict["data"] = self._data.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]
        self._data.load_state_dict(state_dict["data"])


def setup_dataloader(config: PretrainConfig | SFTConfig, micro_batch_size: int) -> tuple[DataLoader, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    if isinstance(config, PretrainConfig):
        dataset = PretrainDataset(config, tokenizer)
    else:  # SFTConfig
        raise ValueError("SFTConfig is not supported")

    workers = config.data.num_workers
    if workers > dataset.num_shards:
        logger.warning(f"num_workers {workers} is >= the dataset shards {dataset.num_shards}")
        workers = dataset.num_shards

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    return dataloader, tokenizer
