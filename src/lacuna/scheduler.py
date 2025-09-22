"""Learning rate scheduler setup."""

from loguru import logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from .config import SchedulerConfig


# inspired by https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/scheduler.py
def setup_scheduler(optimizer: Optimizer, config: SchedulerConfig, total_steps: int) -> LRScheduler:
    """Create learning rate scheduler based on config."""
    warmup_steps = int(config.warmup_ratio * total_steps)
    decay_steps = int(config.decay_ratio * total_steps)

    decay_start_step = total_steps - decay_steps
    constant_steps = decay_start_step - warmup_steps
    logger.info(f"Warmup steps: {warmup_steps}, Constant steps: {constant_steps}, Decay steps: {decay_steps}")

    schedulers = []
    milestones = []

    # Phase 1: Warmup (if any)
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    # Phase 2: Constant (if any)
    if constant_steps > 0:
        constant_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps)
        schedulers.append(constant_scheduler)
        milestones.append(decay_start_step)

    # Phase 3: Final decay (if any)
    if decay_steps > 0:
        if config.decay_type == "linear":
            decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=config.min_lr_ratio, total_iters=decay_steps)
        else:  # cosine
            decay_scheduler = CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min=optimizer.param_groups[0]["lr"] * config.min_lr_ratio
            )
        schedulers.append(decay_scheduler)

    assert len(schedulers) > 0, "No schedulers created, please check your config"
    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers, milestones=milestones)
