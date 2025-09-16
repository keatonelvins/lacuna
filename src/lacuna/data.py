"""Data loading, tokenization, and packing for training."""

from loguru import logger
import multiprocessing as mp
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
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, model_max_length=int(1e10))
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
        self.dp_world, self.dp_rank = get_world_size(), get_rank()  # TODO: needs to use dp_replicate
        self.split = config.data.split

        # TODO: figure out something like accelerate context manager
        if dist.is_initialized() and not is_master():
            dist.barrier()
        try:
            self._dataset = self._build_dataset()
        except Exception as e:
            if is_master():
                dist.barrier()
            raise e
        if dist.is_initialized() and is_master():
            dist.barrier()

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
            drop_last=True,
            pin_memory=True,
            sampler=self.sampler,
        )

    def _load_datasets(self, split: str):
        datasets = []
        for name in self.config.data.datasets:
            if name.startswith("s3://"):
                ds = load_dataset("parquet", data_files=self.config.data.files[name], split=split, num_proc=self.config.data.num_proc)
            else:
                ds = load_dataset(name, split=split, num_proc=self.config.data.num_proc)
            datasets.append(ds)
        return datasets

    def _build_dataset(self):
        """Master process does all hf hub calls and builds dataset. Other processses wait then load from local cache."""
        encode = partial(_encode, tokenizer=get_tokenizer(self.config), column=self.config.data.column)
        pack = partial(pack_bfd, seq_len=self.config.trainer.seq_len)
        raw = self._load_datasets(self.split)
        ds = concatenate_datasets(raw)

        # batch tokenize -> convert to arrow table -> fast bfd packing -> convert to tensors for model forward
        ds = ds.map(
            encode, 
            batched=True, 
            num_proc=self.config.data.num_proc,
            batch_size=self.config.data.map_batch_size, 
            remove_columns=list(next(iter(ds)).keys()),
        ).with_format("arrow")
        ds = ds.map(
            pack, 
            batched=True, 
            batch_size=self.config.data.pack_batch_size, 
            num_proc=self.config.data.num_proc,
        ).with_format("torch")

        self.config.data.fingerprint = ds._fingerprint

        return ds

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    @property
    def length(self) -> int:
        """Return length per epoch of the dataset."""
        return len(self.dataloader)
