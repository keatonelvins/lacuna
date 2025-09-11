"""Data loading, tokenization, and packing for training."""

from datasets import load_dataset, interleave_datasets, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import LacunaConfig
from .distributed import get_rank, get_world_size
from .utils import pack_bfd


class LacunaDataset:
    """Dataset wrapper that supports streaming (iterable) or cached (map-style) datasets."""

    def __init__(self, config: LacunaConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer
        self.dp_world, self.dp_rank = get_world_size(), get_rank()
        self.split = config.data.split

        self._dataset = self._build_dataset()
        if config.data.stream:
            self.sampler = None
        else:
            self.sampler = DistributedSampler(
                self._dataset, num_replicas=self.dp_world, rank=self.dp_rank, shuffle=True, drop_last=True
            )

    def _encode(self, examples):
        if self.config.data.column == "messages":
            results = {"input_ids": [], "assistant_masks": []}
            for messages in examples["messages"]:
                processed = self.tokenizer.apply_chat_template(
                    messages,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                )
                results["input_ids"].append(processed["input_ids"])
                results["assistant_masks"].append(processed["assistant_masks"])
            return results
        else:
            return {"input_ids": self.tokenizer(examples[self.config.data.column]).input_ids}

    def _pack(self, examples):
        return pack_bfd(examples, seq_len=self.config.trainer.seq_len * self.config.trainer.batch_size)

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

        ds = ds.map(
            self._encode, batched=True, batch_size=self.config.data.map_batch_size, remove_columns=[self.config.data.column]
        )
        ds = ds.with_format("arrow").map(self._pack, batched=True, batch_size=self.config.data.pack_batch_size)
        ds = ds.with_format("torch")

        return ds

    def set_epoch(self, epoch: int):
        """Set epoch for proper shuffling across epochs."""
        if self.config.data.stream:
            # TODO: can do self._dataset.set_epoch(epoch) if we catch StopIteration with infinite loop
            # but this won't work with current lr scheduler ratios, should revisit
            raise ValueError("Epoch reseeding is not supported for iterable datasets")
        else:
            self.sampler.set_epoch(epoch)

    def create_dataloader(self, micro_batch_size: int) -> DataLoader:
        """Create the appropriate dataloader for this dataset."""
        if self.config.data.stream:
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
        if not self.config.data.stream and self._dataloader:
            return len(self._dataloader)
        return None


def setup_dataloader(config: LacunaConfig, micro_batch_size: int) -> tuple[DataLoader, PreTrainedTokenizerBase, LacunaDataset]:
    """Setup data pipeline and return dataloader, tokenizer, and dataset."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    dataset = LacunaDataset(config, tokenizer)
    dataloader = dataset.create_dataloader(micro_batch_size)
    return dataloader, tokenizer, dataset
