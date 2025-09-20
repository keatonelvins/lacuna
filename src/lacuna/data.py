"""Data loading, tokenization, and packing for training."""

from loguru import logger
from functools import partial
from torch import distributed as dist
from torch.utils.data import DistributedSampler
from datasets import load_dataset, concatenate_datasets
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import pack_bfd
from .config import LacunaConfig
from .distributed import get_rank, get_world_size, is_master


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
        return {"input_ids": [ids + [tokenizer.eos_token_id] for ids in input_ids]}  # TODO: support other models


def get_tokenizer(config: LacunaConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_override or config.model.name, model_max_length=int(1e10))
    if config.data.chat_template:
        tokenizer.chat_template = config.data.chat_template
    if config.data.eos_token:
        added = tokenizer.add_special_tokens({"eos_token": config.data.eos_token})
        if added > 0:  # TODO: if not in vocab already, need to resize token embeddings (to multiple of 32)
            logger.error(f"{config.data.eos_token} was not already a special token!")

    return tokenizer


class LacunaDataset:
    def __init__(self, config: LacunaConfig):
        self.config = config

        if dist.is_initialized():
            if is_master():
                self._build_dataset()  # warm cache on master first
            dist.barrier()

        self._dataset = self._build_dataset()

        # TODO: use dp_replicate and dp_shard
        self.sampler = DistributedSampler(
            self._dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=True,
            seed=config.trainer.seed,
        )
        self.dataloader = StatefulDataLoader(
            self._dataset,
            drop_last=True,
            pin_memory=True,
            num_workers=self.config.data.num_workers,
            sampler=self.sampler,
        )

    def _load_datasets(self):
        return [
            load_dataset(
                **dataset.model_dump(),
                num_proc=self.config.data.num_proc,
                download_mode="force_redownload" if self.config.data.redownload else None,
            )
            for dataset in self.config.data.datasets
        ]

    def _build_dataset(self):
        """Master process does all hf hub calls and builds dataset. Other processses wait then load from local cache."""
        encode = partial(_encode, tokenizer=get_tokenizer(self.config), column=self.config.data.column)
        pack = partial(pack_bfd, seq_len=self.config.trainer.seq_len)
        ds = concatenate_datasets(self._load_datasets())

        # batch tokenize -> convert to arrow table -> fast bfd packing -> convert to tensors for model forward
        ds = ds.map(
            encode,
            batched=True,
            num_proc=self.config.data.num_proc,
            batch_size=self.config.data.map_bs,
            remove_columns=ds.column_names,
        ).with_format("arrow")
        ds = ds.map(
            pack,
            batched=True,
            batch_size=self.config.data.pack_bs,
            num_proc=self.config.data.num_proc,
            remove_columns=ds.column_names,
        ).with_format("torch")

        return ds

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    @property
    def length(self) -> int:
        return len(self.dataloader)
