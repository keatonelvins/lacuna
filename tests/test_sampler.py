import unittest.mock
from types import SimpleNamespace

from lacuna.config import DataConfig, DatasetConfig, TrainerConfig, ModelConfig
from lacuna.data import PackedDataset


def test_data_parallel_unique_splits():
    """Test that different DP ranks get unique data splits."""
    config = SimpleNamespace(
        model=ModelConfig(),
        data=DataConfig(datasets=[DatasetConfig()], tok_num_proc=1, truncate=True),
        trainer=TrainerConfig(seq_len=128, seed=42, steps=125),
    )

    datasets = []

    for rank in range(4):
        with (
            unittest.mock.patch("lacuna.data.get_world_size", return_value=4),
            unittest.mock.patch("lacuna.data.get_rank", return_value=rank),
        ):
            datasets.append(PackedDataset(config))

    first_samples = []
    for dataset in datasets:
        assert dataset.length == 250, f"Dataset length mismatch: {dataset.length}"
        first_batch = next(iter(dataset.dataloader))
        first_samples.append(tuple(first_batch["input_ids"].flatten().tolist()[:10]))

    assert len(set(first_samples)) == len(first_samples), f"Splits are not unique: {first_samples}"


def test_shuffle_changes_order():
    """Test that shuffle produces different order than sequential."""
    config = SimpleNamespace(
        model=ModelConfig(),
        data=DataConfig(datasets=[DatasetConfig(split="train[:100]")], tok_num_proc=1),
        trainer=TrainerConfig(seq_len=128, seed=42, steps=50),
    )

    dataset = PackedDataset(config)
    first_batch = next(iter(dataset.dataloader))
    first_sample = tuple(first_batch["input_ids"].flatten().tolist()[:20])

    dataset_no_shuffle = PackedDataset(config)
    dataset_no_shuffle.sampler.shuffle = False
    first_batch_no_shuffle = next(iter(dataset_no_shuffle.dataloader))
    first_sample_no_shuffle = tuple(first_batch_no_shuffle["input_ids"].flatten().tolist()[:20])

    assert first_sample != first_sample_no_shuffle


def test_epoch_reshuffling():
    """Test that set_epoch changes sample order across epochs."""
    config = SimpleNamespace(
        model=ModelConfig(),
        data=DataConfig(datasets=[DatasetConfig(split="train[:10]")], tok_num_proc=1, truncate=True),
        trainer=TrainerConfig(seq_len=128, seed=42, steps=5),
    )

    dataset = PackedDataset(config)
    epoch0_samples = [batch["input_ids"].flatten().tolist()[:10] for batch in dataset.dataloader]

    dataset.set_epoch(1)
    epoch1_samples = [batch["input_ids"].flatten().tolist()[:10] for batch in dataset.dataloader]

    assert len(epoch0_samples) == len(epoch1_samples) == 10
    assert sorted(epoch0_samples) == sorted(epoch1_samples)
    assert epoch0_samples != epoch1_samples
