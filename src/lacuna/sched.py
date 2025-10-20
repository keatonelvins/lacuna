"""Learning rate scheduler builder."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from lacuna.config import SchedulerConfig


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig, total_steps: int) -> LRScheduler:
    warmup_steps = int(config.warmup_ratio * total_steps)
    decay_steps = int(config.decay_ratio * total_steps)
    constant_steps = total_steps - warmup_steps - decay_steps

    schedulers, milestones = [], []
    current_step = 0

    # linear warmup (peak * 1e-8 → peak)
    if warmup_steps > 0:
        schedulers.append(LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps))
        current_step += warmup_steps
        milestones.append(current_step)

    # constant stage
    if constant_steps > 0:
        schedulers.append(LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps))
        current_step += constant_steps
        milestones.append(current_step)

    # decay back down (peak → peak * min_lr_ratio)
    if decay_steps > 0:
        if config.decay_type == "cosine":
            schedulers.append(CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=config.end_lr))
        else:  # linear
            schedulers.append(
                LinearLR(optimizer, start_factor=1.0, end_factor=config.end_lr / config.lr, total_iters=decay_steps)
            )

    return schedulers[0] if len(schedulers) == 1 else SequentialLR(optimizer, schedulers, milestones=milestones)
