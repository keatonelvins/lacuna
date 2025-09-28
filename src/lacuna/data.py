"""Data loading, tokenization, and packing for training."""

from functools import partial
from torch import distributed as dist
from torch.utils.data import DistributedSampler
from datasets import load_dataset, concatenate_datasets
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import pack_bfd
from .config import LacunaConfig, DatasetConfig
from .distributed import get_rank, get_world_size, is_master, get_dp_params


def _encode(examples, tokenizer, column):
    if column == "messages":
        out = tokenizer.apply_chat_template(
            examples["messages"],
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        return {"input_ids": out["input_ids"], "assistant_masks": out["assistant_masks"]}
    else:
        assert "qwen" in tokenizer.name_or_path.lower(), "need to check if glm uses bos"
        input_ids = tokenizer(examples[column]).input_ids
        return {"input_ids": [ids + [tokenizer.eos_token_id] for ids in input_ids]}


def get_tokenizer(config: LacunaConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_override or config.model.name, model_max_length=int(1e10))
    if config.data.chat_template:
        tokenizer.chat_template = config.data.chat_template
    if config.data.eos_token:
        added = tokenizer.add_special_tokens({"eos_token": config.data.eos_token})
        assert added == 0, "eos token was not already a special token! resizing unsupported atm"

    return tokenizer


class LacunaDataset:
    def __init__(self, config: LacunaConfig, datasets: list[DatasetConfig] | None = None):
        self.config = config
        self.datasets = datasets or self.config.data.datasets

        if dist.is_initialized():
            if is_master():
                self._build_dataset()  # warm cache on master first
            dist.barrier()

        self._dataset = self._build_dataset()

        num_replicas, rank = get_world_size(), get_rank()
        dp_replicate, dp_shard = get_dp_params(config)

        if dp_replicate > 1 and dp_shard > 1:
            num_replicas = dp_replicate
            rank = rank // dp_shard

        self.sampler = DistributedSampler(
            self._dataset,
            num_replicas=num_replicas,
            rank=rank,
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
            for dataset in self.datasets
        ]

    def _build_dataset(self):
        """Master process does all hf hub calls and builds dataset first. Others wait then load from local cache."""
        tokenizer = get_tokenizer(self.config)
        ds = concatenate_datasets(self._load_datasets())
        cfg = self.config.data

        # batch tokenize -> convert to arrow table -> fast bfd packing -> convert to tensors for model forward
        ds = ds.map(
            partial(_encode, tokenizer=tokenizer, column=cfg.column),
            batched=True,
            num_proc=cfg.num_proc,
            batch_size=cfg.map_bs,
            remove_columns=ds.column_names,
        ).with_format("arrow")
        ds = ds.map(
            partial(pack_bfd, seq_len=self.config.trainer.seq_len, context_len=cfg.context_len, truncate=cfg.truncate),
            batched=True,
            batch_size=cfg.pack_bs,
            num_proc=cfg.num_proc,
            remove_columns=ds.column_names,
        ).with_format("torch")

        return ds

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    @property
    def length(self) -> int:
        return len(self.dataloader)
