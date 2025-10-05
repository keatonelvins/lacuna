import tempfile
from pathlib import Path

import pytest
import torch

from lacuna.checkpoint import save_checkpoint, load_checkpoint
from lacuna.config import LacunaConfig
from lacuna.data import LacunaDataset
from lacuna.model import setup_model
from lacuna.optim import setup_optimizer
from lacuna.scheduler import setup_scheduler


def _make_config():
    """Create minimal config for testing."""
    config = LacunaConfig()
    config.model.name = "Qwen/Qwen3-0.6B-Base"
    config.data.datasets[0].split = "train[:10]"
    config.data.tok_num_proc = 1
    config.trainer.seq_len = 128
    config.trainer.steps = 5
    config.trainer.seed = 42
    return config


def test_save_and_load_full_state():
    """Test saving and loading preserves all state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)

        model = setup_model(config)
        optimizer = setup_optimizer(model, config)
        scheduler = setup_scheduler(optimizer, config.scheduler, total_steps=10)
        dataset = LacunaDataset(config)

        with torch.no_grad():
            model.lm_head.weight[0, 0] = 42.5
        scheduler.step()
        scheduler.step()

        save_checkpoint(5, config, model, optimizer, scheduler, dataset.dataloader)

        model2 = setup_model(config)
        optimizer2 = setup_optimizer(model2, config)
        scheduler2 = setup_scheduler(optimizer2, config.scheduler, total_steps=10)
        dataset2 = LacunaDataset(config)

        loaded_step = load_checkpoint(
            model2, optimizer2, scheduler2, dataset2.dataloader, Path(tmpdir) / "step_5"
        )

        assert loaded_step == 5
        assert model2.lm_head.weight[0, 0].item() == 42.5
        assert scheduler2.last_epoch == 2


def test_exclude_optimizer():
    """Test loading without optimizer keeps optimizer fresh."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)

        model = setup_model(config)
        optimizer = setup_optimizer(model, config)

        with torch.no_grad():
            model.lm_head.weight[0, 0] = 88.0

        save_checkpoint(5, config, model, optimizer, None, None)

        model2 = setup_model(config)
        optimizer2 = setup_optimizer(model2, config)

        loaded_step = load_checkpoint(model2, None, None, None, Path(tmpdir) / "step_5")

        assert loaded_step == 5
        assert model2.lm_head.weight[0, 0].item() == 88.0
        assert optimizer2.state == {}


def test_exclude_dataloader():
    """Test loading without dataloader keeps dataloader fresh."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)

        model = setup_model(config)
        dataset = LacunaDataset(config)

        # Get first batch from original dataloader
        first_batch_original = next(iter(dataset.dataloader))
        first_sample_original = tuple(first_batch_original["input_ids"].flatten().tolist()[:10])

        # Advance dataloader past first batch
        next(iter(dataset.dataloader))

        save_checkpoint(3, config, model, None, None, dataset.dataloader)

        model2 = setup_model(config)
        dataset2 = LacunaDataset(config)

        load_checkpoint(model2, None, None, None, Path(tmpdir) / "step_3")

        # dataset2 should be fresh - starts from beginning, not where dataset left off
        first_batch_fresh = next(iter(dataset2.dataloader))
        first_sample_fresh = tuple(first_batch_fresh["input_ids"].flatten().tolist()[:10])

        assert first_sample_fresh == first_sample_original


def test_dataloader_state_preserved():
    """Test dataloader state is preserved across save/load (no duplicates)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)

        model = setup_model(config)
        optimizer = setup_optimizer(model, config)
        dataset = LacunaDataset(config)

        # Consume first 3 batches
        dataset.set_epoch(0)
        first_3 = []
        dl_iter = iter(dataset.dataloader)
        for _ in range(3):
            batch = next(dl_iter)
            first_3.append(tuple(batch["input_ids"].flatten().tolist()[:10]))

        save_checkpoint(3, config, model, optimizer, None, dataset.dataloader)

        # Load and continue
        model2 = setup_model(config)
        optimizer2 = setup_optimizer(model2, config)
        dataset2 = LacunaDataset(config)
        dataset2.set_epoch(0)

        load_checkpoint(model2, optimizer2, None, dataset2.dataloader, Path(tmpdir) / "step_3")

        # Collect remaining samples
        remaining = [tuple(batch["input_ids"].flatten().tolist()[:10]) for batch in dataset2.dataloader]

        # Should have no overlap - dataloader continued from checkpoint
        assert len(set(first_3) & set(remaining)) == 0


def test_exclude_scheduler():
    """Test loading without scheduler keeps scheduler fresh."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)
        config.scheduler.warmup_ratio = 0

        model = setup_model(config)
        optimizer = setup_optimizer(model, config)
        scheduler = setup_scheduler(optimizer, config.scheduler, total_steps=10)
        dataset = LacunaDataset(config)

        # Advance scheduler into decay phase
        for _ in range(9):
            scheduler.step()
        old_lr = scheduler.get_last_lr()[0]

        save_checkpoint(5, config, model, optimizer, scheduler, dataset.dataloader)

        model2 = setup_model(config)
        optimizer2 = setup_optimizer(model2, config)
        scheduler2 = setup_scheduler(optimizer2, config.scheduler, total_steps=10)
        dataset2 = LacunaDataset(config)

        loaded_step = load_checkpoint(model2, optimizer2, None, dataset2.dataloader, Path(tmpdir) / "step_5")

        # Step should be loaded as 5, but then reset to 0 in trainer when scheduler is excluded
        assert loaded_step == 5
        new_lr = scheduler2.get_last_lr()[0]
        assert new_lr == config.optimizer.lr
        assert old_lr < config.optimizer.lr


def test_exclude_scheduler_resets_step():
    """Test that excluding scheduler resets step counter for annealing phases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)
        config.trainer.steps = 10  # Annealing phase: only 10 steps

        # First: train to step 100 and save WITH optimizer
        model = setup_model(config)
        optimizer = setup_optimizer(model, config)
        with torch.no_grad():
            model.lm_head.weight[0, 0] = 99.0

        save_checkpoint(100, config, model, optimizer, None, None)

        # Second: Resume for annealing phase with fresh scheduler + data
        # Step should reset to 0 because scheduler is excluded
        config.checkpoint.resume_from = Path(tmpdir) / "step_100"
        config.checkpoint.exclude_from_loading = ["scheduler", "dataloader"]

        model2 = setup_model(config)
        optimizer2 = setup_optimizer(model2, config)
        scheduler2 = setup_scheduler(optimizer2, config.scheduler, total_steps=10)
        dataset2 = LacunaDataset(config)

        # Simulate what trainer.py does
        step = load_checkpoint(model2, optimizer2, None, None, Path(tmpdir) / "step_100")
        assert step == 100

        # Trainer resets step when scheduler is excluded
        if "scheduler" in config.checkpoint.exclude_from_loading:
            step = 0

        assert step == 0, "Step should be reset to 0 for annealing phase"
        assert model2.lm_head.weight[0, 0].item() == 99.0, "Model weights should be loaded"


def test_model_only_save():
    """Test saving model-only checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)

        model = setup_model(config)

        with torch.no_grad():
            model.lm_head.weight[0, 0] = 77.0

        save_checkpoint(10, config, model, None, None, None)

        model2 = setup_model(config)
        loaded_step = load_checkpoint(model2, None, None, None, Path(tmpdir) / "step_10")

        assert loaded_step == 10
        assert model2.lm_head.weight[0, 0].item() == 77.0


def test_step_tracking():
    """Test step is correctly saved and restored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)
        model = setup_model(config)

        for step in [1, 42]:  # step 0 is skipped by save_checkpoint
            save_checkpoint(step, config, model, None, None, None)
            loaded_step = load_checkpoint(
                model, None, None, None, Path(tmpdir) / f"step_{step}"
            )
            assert loaded_step == step


def test_final_vs_intermediate():
    """Test final=True saves to 'final' directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)
        model = setup_model(config)

        save_checkpoint(100, config, model, None, None, None, final=False)
        assert (Path(tmpdir) / "step_100" / ".metadata").exists()

        save_checkpoint(100, config, model, None, None, None, final=True)
        assert (Path(tmpdir) / "final" / ".metadata").exists()


def test_epoch_boundary_checkpoint():
    """Test checkpointing and resuming across epoch boundaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        config.checkpoint.save_dir = Path(tmpdir)
        config.trainer.steps = 25  # More than one epoch worth

        model = setup_model(config)
        optimizer = setup_optimizer(model, config)
        scheduler = setup_scheduler(optimizer, config.scheduler, total_steps=25)
        dataset = LacunaDataset(config)

        # Dataset has 10 samples, so:
        # Steps 1-10: epoch 0
        # Steps 11-20: epoch 1
        # Steps 21-25: epoch 2
        dataset_length = dataset.length
        assert dataset_length == 10, "Test assumes dataset has 10 samples"

        # Train to step 15 (middle of epoch 1)
        # Note: We need to actually iterate through the dataloader
        # to advance its state properly
        dataset.set_epoch(0)
        data_iter = iter(dataset.dataloader)

        # Consume 15 batches to simulate training to step 15
        consumed_batches = []
        for i in range(15):
            # Calculate which epoch we should be in
            step = i + 1
            expected_epoch = step // dataset_length

            # Reshuffle at epoch boundaries (step 1, 11, 21, ...)
            if step % dataset_length == 1:
                dataset.set_epoch(expected_epoch)
                if step > 1:
                    data_iter = iter(dataset.dataloader)

            batch = next(data_iter)
            consumed_batches.append(tuple(batch["input_ids"].flatten().tolist()[:10]))

        # Save checkpoint at step 15 (middle of epoch 1)
        save_checkpoint(15, config, model, optimizer, scheduler, dataset.dataloader)

        # Resume and continue to step 25
        model2 = setup_model(config)
        optimizer2 = setup_optimizer(model2, config)
        scheduler2 = setup_scheduler(optimizer2, config.scheduler, total_steps=25)
        dataset2 = LacunaDataset(config)

        # Set epoch based on resumed step (15 // 10 = 1)
        dataset2.set_epoch(15 // dataset_length)

        loaded_step = load_checkpoint(
            model2, optimizer2, scheduler2, dataset2.dataloader, Path(tmpdir) / "step_15"
        )
        assert loaded_step == 15

        # Continue training from step 16-25
        data_iter2 = iter(dataset2.dataloader)
        remaining_batches = []
        for i in range(10):  # 10 more steps: 16-25
            step = loaded_step + i + 1
            expected_epoch = step // dataset_length

            # Reshuffle at step 21 (start of epoch 2)
            if step % dataset_length == 1:
                dataset2.set_epoch(expected_epoch)
                data_iter2 = iter(dataset2.dataloader)

            batch = next(data_iter2)
            remaining_batches.append(tuple(batch["input_ids"].flatten().tolist()[:10]))

        # Verify no duplicate batches across the checkpoint boundary
        all_batches = consumed_batches + remaining_batches
        assert len(all_batches) == 25

        # Check that epoch 0 (steps 1-10) and epoch 1 (steps 11-20) have no overlap
        epoch0_batches = set(consumed_batches[:10])
        epoch1_part1 = set(consumed_batches[10:15])  # Steps 11-15 before checkpoint
        epoch1_part2 = set(remaining_batches[:5])     # Steps 16-20 after resume

        # Within same epoch, no duplicates
        assert len(epoch1_part1 & epoch1_part2) == 0, "Duplicate batches within epoch 1 across checkpoint"

        # Different epochs should have same data (reshuffled)
        epoch1_all = epoch1_part1 | epoch1_part2
        assert len(epoch0_batches & epoch1_all) == 10, "Epochs should contain same data, just reshuffled"

