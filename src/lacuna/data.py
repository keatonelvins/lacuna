"""Data loading, tokenization, and packing for training."""

from datasets import load_dataset, interleave_datasets, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import PretrainConfig, SFTConfig
from .distributed import get_rank, get_world_size
from .utils import pack_bfd


class LacunaDataset:
    """Unified dataset wrapper for both iterable and map-style pipelines."""

    def __init__(self, config: PretrainConfig | SFTConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer
        self.dp_world, self.dp_rank = get_world_size(), get_rank()
        self.split = config.data.split

        if config.data.iterable:
            self._dataset = self._build_iterable()
            self.sampler = None
        else:
            self._dataset = self._build_mapstyle()
            self.sampler = DistributedSampler(
                self._dataset, num_replicas=self.dp_world, rank=self.dp_rank, shuffle=True, drop_last=True
            )

    def _encode(self, examples):
        out = self.tokenizer(examples["text"], add_special_tokens=False, truncation=False, padding=False)
        return {"input_ids": [ids + [self.tokenizer.eos_token_id] for ids in out["input_ids"]]}

    def _pack(self, examples):
        return pack_bfd(examples, seq_length=self.config.data.seq_len * self.config.trainer.batch_size)

    def _build_iterable(self):
        if self.config.data.stream:
            raw = [load_dataset(name, split=self.split, streaming=True) for name in self.config.data.datasets]
        else:
            raw = [load_dataset(name, split=self.split).to_iterable_dataset() for name in self.config.data.datasets]

        ds = interleave_datasets(
            raw,
            probabilities=self.config.data.sampling_probs,
            stopping_strategy="first_exhausted",
            seed=self.config.data.seed,
        )

        ds = ds.shuffle(seed=self.config.data.seed, buffer_size=self.config.data.shuffle_buffer)
        ds = split_dataset_by_node(ds, rank=self.dp_rank, world_size=self.dp_world)
        ds = ds.map(self._encode, batched=True, batch_size=self.config.data.map_batch_size, remove_columns=["text"])
        ds = ds.with_format("arrow").map(self._pack, batched=True, batch_size=self.config.data.pack_batch_size)
        ds = ds.with_format("torch")

        return ds

    def _build_mapstyle(self):
        raw = [load_dataset(name, split=self.split) for name in self.config.data.datasets]
        ds = concatenate_datasets(raw)
        ds = ds.map(self._encode, batched=True, remove_columns=["text"])
        ds = ds.map(self._pack, batched=True, batch_size=self.config.data.pack_batch_size)
        ds = ds.with_format("torch")

        return ds

    def set_epoch(self, epoch: int):
        """Set epoch for proper shuffling across epochs."""
        if self.config.data.iterable:
            # TODO: can do self._dataset.set_epoch(epoch) if we catch StopIteration with infinite loop
            # but this won't work with current lr scheduler ratios, should revisit
            raise ValueError("Epoch reseeding is not supported for iterable datasets")
        else:
            self.sampler.set_epoch(epoch)

    def create_dataloader(self, micro_batch_size: int) -> DataLoader:
        """Create the appropriate dataloader for this dataset."""
        if self.config.data.iterable:
            self._dataloader = StatefulDataLoader(
                self._dataset, num_workers=self.config.data.num_workers, drop_last=True, pin_memory=True, persistent_workers=True
            )
        else:
            self._dataloader = DataLoader(
                self._dataset,
                sampler=self.sampler,
                num_workers=self.config.data.num_workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=True,
            )

        return self._dataloader

    @property
    def length(self) -> int | None:
        """Return length per epoch of the dataset."""
        if not self.config.data.iterable and self._dataloader:
            return len(self._dataloader)
        return None


def setup_dataloader(
    config: PretrainConfig | SFTConfig, micro_batch_size: int
) -> tuple[DataLoader, PreTrainedTokenizerBase, LacunaDataset]:
    """Setup data pipeline and return dataloader, tokenizer, and dataset."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    dataset = LacunaDataset(config, tokenizer)
    dataloader = dataset.create_dataloader(micro_batch_size)
    return dataloader, tokenizer, dataset
