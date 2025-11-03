"""Data loading and preprocessing."""

import torch
import numpy as np
import pyarrow as pa
from loguru import logger
import pyarrow.compute as pc
from functools import partial
from collections import defaultdict, deque
from torch.utils.data import DistributedSampler
from datasets import load_dataset, concatenate_datasets
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lacuna.config import TrainConfig
from lacuna.utils import get_rank, is_master, get_world_size, run_master_first


def encode(examples, tokenizer, column, template=False) -> dict:
    """Batch tokenize input text."""
    if template:
        return tokenizer.apply_chat_template(
            examples[column],
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    else:
        assert tokenizer.bos_token is None, "bos token is not supported atm"
        input_ids = tokenizer(examples[column]).input_ids
        return {"input_ids": [ids + [tokenizer.eos_token_id] for ids in input_ids]}


def get_tokenizer(config: TrainConfig) -> PreTrainedTokenizerBase:
    """Get tokenizer object and update with config values."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, model_max_length=int(1e10))
    if config.data.chat_template:
        tokenizer.chat_template = config.data.chat_template
    if config.data.eos_token:
        added = tokenizer.add_special_tokens({"eos_token": config.data.eos_token})
        assert added == 0, "eos token was not already a special token! resizing unsupported atm"

    return tokenizer


def build_inputs(batch: dict[str, torch.Tensor], pad_id: int, pad_to: int) -> dict[str, torch.Tensor]:
    """Pad tensors to multiple of pad_to and build labels column w/ appropriate masking."""
    labels = batch["input_ids"].clone()
    labels[batch["position_ids"] == 0] = -100  # mask boundary tokens
    if "assistant_masks" in batch:
        labels[batch["assistant_masks"] == 0] = -100  # mask non-assistant tokens

    batch["input_ids"] = pad(batch["input_ids"], pad_id=pad_id, pad_to=pad_to)
    batch["position_ids"] = pad(batch["position_ids"], pad_id=0, pad_to=pad_to)
    batch["labels"] = pad(labels, pad_id=-100, pad_to=pad_to)

    return batch


class PackedDataset:
    """Padding-free packed dataset class for training."""

    def __init__(self, config: TrainConfig, mesh=None, train: bool = True):
        self.config = config
        self.datasets = config.data.datasets if train else config.evals.datasets

        with run_master_first():
            self._dataset = self._build_dataset(skip_cache=self.config.data.skip_cache and is_master())

        if mesh is not None and mesh.ndim > 1:
            dp_mesh = mesh["dp"]
            num_replicas, rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            num_replicas, rank = get_world_size(), get_rank()

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
        self.set_epoch(0)

    def _load_datasets(self, skip_cache: bool = False):
        return [
            load_dataset(
                **dataset.model_dump(),
                num_proc=self.config.data.tok_num_proc,
                download_mode="force_redownload" if skip_cache else None,
            )
            for dataset in self.datasets
        ]

    def _build_dataset(self, skip_cache: bool = False):
        """Concat datasets, tokenize, pack, and convert to tensors. Will load from local cache if available."""
        tokenizer = get_tokenizer(self.config)
        cfg = self.config.data
        datasets = [
            ds.map(
                partial(encode, tokenizer=tokenizer, column=cfg.column, template=cfg.chat_template is not None),
                desc=f"Tokenizing data with (bs={cfg.tok_bs})",
                batched=True,
                batch_size=cfg.tok_bs,
                num_proc=cfg.tok_num_proc,
                remove_columns=ds.column_names,
            )
            for ds in self._load_datasets(skip_cache=skip_cache)
        ]

        # batch tokenize -> convert to arrow table -> fast bfd packing -> convert to tensors for model forward
        ds = concatenate_datasets(datasets).with_format("arrow")
        ds = ds.map(
            partial(pack, seq_len=self.config.trainer.seq_len, context_len=cfg.context_len, truncate=cfg.truncate),
            desc=f"Packing data with (bs={cfg.pack_bs})",
            batched=True,
            batch_size=cfg.pack_bs,
            num_proc=cfg.pack_num_proc or (len(ds) // cfg.pack_bs) + 1,
            remove_columns=ds.column_names,
        ).with_format("torch")
        ds = ds.map(
            partial(build_inputs, pad_id=tokenizer.pad_token_id, pad_to=self.config.trainer.pad_to),
            desc=f"Building inputs with (pad_to={self.config.trainer.pad_to})",
            num_proc=cfg.tok_num_proc,
            remove_columns=ds.column_names,
        )

        return ds

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    @property
    def length(self) -> int:
        return len(self.dataloader)


class IntSucc:
    """Find next greater integer in a set of integers."""

    __slots__ = ("N", "bits")

    def __init__(self, maxval: int):
        assert maxval >= 1
        self.N, self.bits = maxval, 0

    def add(self, i: int):
        self.bits |= 1 << (i - 1)

    def discard(self, i: int):
        self.bits &= ~(1 << (i - 1))

    def next_geq(self, x: int) -> int:
        y = self.bits >> (x - 1)
        assert y, "no successor present (missing sentinel?)"
        return x + ((y & -y).bit_length() - 1)


def take(arr, idx):
    """Take elements from a pyarrow array based on indices."""
    idx = np.asarray(idx, dtype=np.int32)
    out = pc.take(arr, pa.array(idx, type=pa.int32()))
    return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out


def pack(examples: pa.Table, seq_len: int, context_len: int | None = None, truncate: bool = True) -> pa.Table:
    """Pack examples into sequences up to seq_len. If example is > context_len, either drop or truncate it."""
    has_masks = "assistant_masks" in examples.column_names
    context_len = context_len or seq_len  # default to seq_len if context_len is not provided

    if truncate:
        ids = pc.list_slice(examples["input_ids"], 0, context_len)
        masks = pc.list_slice(examples["assistant_masks"], 0, context_len) if has_masks else None
    else:  # drop samples longer than context_len
        sample_lens = pc.list_value_length(examples["input_ids"])
        long_sample_mask = pc.less_equal(sample_lens, context_len)
        num_kept, num_total = pc.sum(long_sample_mask).as_py(), len(sample_lens)
        logger.info(f"Dropped {num_total - num_kept} overlong examples (len > {context_len})")
        ids = pc.filter(examples["input_ids"], long_sample_mask)
        masks = pc.filter(examples["assistant_masks"], long_sample_mask) if has_masks else None

    lens = pc.list_value_length(ids).to_numpy()
    order = np.argsort(-lens)

    succ = IntSucc(seq_len)
    succ.add(seq_len)  # sentinel enables new bins
    by_space = defaultdict(deque)  # space -> deque[bins]
    bins = []  # each: {"ids": [...], "len": int}

    for i in order:
        L = int(lens[i])
        if not L:
            continue
        s = succ.next_geq(L)
        b = by_space[s].popleft() if s < seq_len else {"ids": [], "len": 0}
        if s < seq_len and not by_space[s]:
            succ.discard(s)
        b["ids"].append(int(i))
        b["len"] += L
        if s == seq_len:
            bins.append(b)
        ns = s - L
        by_space[ns].append(b)
        if ns:
            succ.add(ns)

    reorder = [j for b in bins for j in b["ids"]]
    ids_taken = take(ids, reorder)
    if has_masks:
        masks_taken = take(masks, reorder)

    # offsets (match ListArray vs LargeListArray via dtype)
    tok_counts = [b["len"] for b in bins]
    odtype = ids_taken.offsets.type.to_pandas_dtype()
    offs = np.cumsum([0] + tok_counts, dtype=odtype)

    LA = type(ids_taken)
    packed_ids = LA.from_arrays(offs, ids_taken.values)

    # position_ids: reset to 0 at each original example boundary
    dl = lens[reorder]
    T = int(offs[-1])
    pos = np.ones(T, dtype=np.int32)
    pos[0] = 0
    if dl.size > 1:
        cut = dl[:-1].cumsum()
        pos[cut] = -(dl[:-1] - 1)
    pos = pos.cumsum()
    position_ids = LA.from_arrays(offs, pa.array(pos, type=pa.int32()))

    if has_masks:
        packed_masks = LA.from_arrays(offs, masks_taken.values)
        return pa.Table.from_arrays(
            [packed_ids, position_ids, packed_masks], names=["input_ids", "position_ids", "assistant_masks"]
        )
    return pa.Table.from_arrays([packed_ids, position_ids], names=["input_ids", "position_ids"])


def pad(t: torch.Tensor, pad_id: int, pad_to: int) -> torch.Tensor:
    """Pad a 1D tensor to the next multiple of pad_to."""
    remainder = t.size(0) % pad_to
    if remainder == 0:
        return t
    pad_len = pad_to - remainder
    return torch.cat((t, torch.full((pad_len,), pad_id, dtype=t.dtype, device=t.device)))
