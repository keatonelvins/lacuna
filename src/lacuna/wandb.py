from typing import Any

import wandb

from .config import LacunaConfig
from .distributed import get_rank


def init_wandb(config: LacunaConfig) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.enabled or get_rank() != 0:
        return None

    config_dict = config.model_dump()

    run = wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        entity=config.wandb.entity,
        config=config_dict,
        mode="offline" if config.wandb.offline else "online",
    )

    return run


def log_metrics(
    metrics: dict[str, Any],
    step: int,
    run: wandb.sdk.wandb_run.Run | None,
) -> None:
    if run:
        wandb.log(metrics, step=step)


def finish(run: wandb.sdk.wandb_run.Run | None) -> None:
    if run:
        wandb.finish()
