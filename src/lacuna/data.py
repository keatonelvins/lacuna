"""Data loading, tokenization, and packing for training."""

from loguru import logger
import multiprocessing as mp
from functools import partial
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import pack_bfd
from .config import LacunaConfig
from .distributed import get_rank, get_world_size


def _encode(examples, tokenizer, column):
    if column == "messages":
        out = tokenizer.apply_chat_template(
            examples["messages"],
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        return {"input_ids": out["input_ids"], "assistant_masks": out["assistant_masks"]}
    else:
        input_ids = tokenizer(examples[column]).input_ids
        return {"input_ids": [ids + [tokenizer.eos_token_id] for ids in input_ids]}


def get_tokenizer(config: LacunaConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if config.data.chat_template:
        tokenizer.chat_template = config.data.chat_template
    if config.data.eos_token:
        added = tokenizer.add_special_tokens({"eos_token": config.data.eos_token})
        if added > 0:  # TODO: if not in vocab already, need to resize token embeddings (to multiple of 32)
            logger.error(f"{config.data.eos_token} was not already a special token!")

    return tokenizer


class LacunaDataset:
    """Dataset wrapper that supports streaming (iterable) or cached (map-style) datasets."""

    def __init__(self, config: LacunaConfig):
        self.config = config
        self.dp_world, self.dp_rank = get_world_size(), get_rank()
        self.split = config.data.split

        self._dataset = self._build_dataset()
        self.config.data.fingerprint = self._dataset._fingerprint
        self.sampler = None
        if not config.data.stream:
            self.sampler = DistributedSampler(
                self._dataset,
                num_replicas=self.dp_world,
                rank=self.dp_rank,
                shuffle=True,
                drop_last=True,
                seed=config.trainer.seed,
            )

        self.dataloader = StatefulDataLoader(
            self._dataset,
            num_workers=self.config.data.num_workers,
            drop_last=True,
            pin_memory=True,
            multiprocessing_context=mp.get_context("spawn"),
            sampler=self.sampler,
        )
        self._current_iter = None

    def __next__(self):
        if self._current_iter is None:
            self._current_iter = iter(self.dataloader)
        try:
            return next(self._current_iter)
        except StopIteration:
            self._current_iter = iter(self.dataloader)
            return next(self._current_iter)

    def _load_datasets(self, split: str, stream: bool):
        datasets = []
        for name in self.config.data.datasets:
            if name.startswith("s3://"):
                ds = load_dataset("parquet", data_files=self.config.data.files[name], split=split, streaming=stream)
            else:
                ds = load_dataset(name, split=split, streaming=stream)

            datasets.append(ds)
        return datasets

    def _build_dataset(self):
        raw = self._load_datasets(self.split, self.config.data.stream)
        if self.config.data.stream:
            ds = interleave_datasets(
                raw,
                probabilities=self.config.data.sampling_probs,
                stopping_strategy="first_exhausted",
                seed=self.config.trainer.seed,
            ).shuffle(seed=self.config.trainer.seed, buffer_size=self.config.data.shuffle_buffer)
            ds = split_dataset_by_node(ds, rank=self.dp_rank, world_size=self.dp_world)
        else:
            ds = concatenate_datasets(raw)

        encode = partial(_encode, tokenizer=get_tokenizer(self.config), column=self.config.data.column)
        pack = partial(pack_bfd, seq_len=self.config.trainer.seq_len)

        # batch tokenize -> convert to arrow table -> fast bfd packing -> convert to tensors for model forward
        ds = ds.map(encode, batched=True, batch_size=self.config.data.map_batch_size, remove_columns=[self.config.data.column])
        ds = ds.with_format("arrow").map(pack, batched=True, batch_size=self.config.data.pack_batch_size)
        ds = ds.with_format("torch")

        return ds

    def set_epoch(self, epoch: int):
        """Set epoch for proper shuffling across epochs."""
        if self.config.data.stream:
            self._dataset.set_epoch(epoch)
        else:
            self.sampler.set_epoch(epoch)

    @property
    def length(self) -> int:
        """Return length per epoch of the dataset."""
        if self.config.data.stream:
            return self.config.trainer.steps
        return len(self.dataloader) if self.dataloader else 1


def setup_dataloader(config: LacunaConfig) -> tuple[StatefulDataLoader, LacunaDataset]:
    dataset = LacunaDataset(config)
    return dataset.dataloader, dataset
