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
from .utils import pack_bfd

def _get_iterable_dataset(config: PretrainConfig | SFTConfig) -> IterableDataset:
    """Get single IterableDataset from merged hf datasets."""
    loader = partial(load_dataset, split=config.data.split, streaming=config.data.stream)
    datasets = [loader(name) for name in config.data.datasets]

    if not config.data.stream:
        datasets = [dataset.to_iterable_dataset(num_shards=get_world_size()) for dataset in datasets]

    dataset = interleave_datasets(datasets, stopping_strategy="all_exhausted", probabilities=config.data.sampling_probs)

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
        self.seq_len = config.data.seq_len

        logger.info(f"Loading datasets: {config.data.datasets}")
        dataset = _get_iterable_dataset(config)
        self._data = split_dataset_by_node(dataset, get_rank(), get_world_size())
        self._data = self._data.map(self._encode, batched=True, batch_size=5000, remove_columns=["text"])
        self._data = self._data.with_format("arrow")
        self._data = self._data.map(self._pack, batched=True, batch_size=5000)

        self.num_shards = self._data.num_shards

    def _encode(self, examples):
        out = self.tokenizer(examples["text"], add_special_tokens=False, truncation=False, padding=False)
        return {"input_ids": [ids + [self.tokenizer.eos_token_id] for ids in out["input_ids"]]}

    def _pack(self, examples):
        return pack_bfd(examples, seq_length=self.seq_len)

    def __iter__(self):
        for packed_batch in self._data:
            for i in range(packed_batch.num_rows):
                input_ids = packed_batch["input_ids"][i].as_py()
                position_ids = packed_batch["position_ids"][i].as_py()

                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "position_ids": torch.tensor(position_ids, dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long)
                }

    def state_dict(self):
        return {"data": self._data.state_dict()}

    def load_state_dict(self, state_dict):
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
