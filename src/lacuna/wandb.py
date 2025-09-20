"""Weights and Biases logging."""

import wandb

from .config import LacunaConfig
from .utils import master_only


@master_only
def init_wandb(config: LacunaConfig) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.project:
        return None

    run = wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        entity=config.wandb.entity,
        config=config.model_dump(mode="json"),
        mode="offline" if config.wandb.offline else "online",
    )

    return run


@master_only
def log_wandb_metrics(
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    run: wandb.sdk.wandb_run.Run | None,
) -> None:
    if run:
        metrics = {
            "train/loss": loss,
            "train/lr": lr,
            "train/grad_norm": grad_norm,
        }
        wandb.log(metrics, step=step)


@master_only
def finish(run: wandb.sdk.wandb_run.Run | None) -> None:
    if run:
        wandb.finish()
