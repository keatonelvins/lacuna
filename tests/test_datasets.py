import unittest.mock
from types import SimpleNamespace

from lacuna.config import ModelConfig, DataConfig, DatasetConfig, TrainerConfig
from lacuna.data import LacunaDataset

def test_dataset_cache_reuse():
    """Test that LacunaDataset reuses cache when built with same config."""
    config = SimpleNamespace(
        model=ModelConfig(),
        data=DataConfig(datasets=[DatasetConfig(split="train[:10]")], num_proc=1),
        trainer=TrainerConfig(seq_len=128, seed=42),
    )

    dataset1 = LacunaDataset(config)
    fingerprint1 = dataset1._dataset._fingerprint
    dataset2 = LacunaDataset(config)
    fingerprint2 = dataset2._dataset._fingerprint

    assert fingerprint1 == fingerprint2
    assert len(dataset1._dataset) == len(dataset2._dataset) == 10

    config.trainer.seq_len = 512
    dataset3 = LacunaDataset(config)
    fingerprint3 = dataset3._dataset._fingerprint

    assert fingerprint1 != fingerprint3
    assert len(dataset3._dataset) == 5


def test_data_parallel_unique_splits():
    """Test that different DP ranks get unique data splits."""
    config = SimpleNamespace(
        model=ModelConfig(),
        data=DataConfig(datasets=[DatasetConfig()], num_proc=1),
        trainer=TrainerConfig(seq_len=128, seed=42, steps=125),
    )

    datasets = []

    for rank in range(8):
        with (
            unittest.mock.patch("lacuna.data.get_world_size", return_value=8),
            unittest.mock.patch("lacuna.data.get_rank", return_value=rank),
        ):
            datasets.append(LacunaDataset(config))

    first_samples = []
    for dataset in datasets:
        assert dataset.length == 125, f"Dataset length mismatch: {dataset.length}"
        first_batch = next(iter(dataset.dataloader))
        first_samples.append(tuple(first_batch["input_ids"].flatten().tolist()[:10]))

    assert len(set(first_samples)) == len(first_samples), f"Splits are not unique: {first_samples}"
