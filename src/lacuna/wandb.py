from typing import Any

import wandb

from .config import LacunaConfig
from .distributed import is_master
from .metrics import StateTracker


def init_wandb(config: LacunaConfig) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.enabled or not is_master():
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


def prepare_wandb_metrics(
    loss: float,
    grad_norm: float,
    lr: float,
    redline_metrics: dict[str, float],
    state: StateTracker,
) -> dict[str, Any]:
    """Prepare metrics for wandb logging."""
    metrics = {
        "train/loss": loss,
        "train/grad_norm": grad_norm,
        "train/lr": lr,
        "train/total_tokens": state.total_tokens,
        "perf/data_loading_pct": redline_metrics.get("data_pct", 0.0),
        "perf/tps": redline_metrics.get("tps", 0.0),
        "perf/tflops": redline_metrics.get("tflops", 0.0),
        "perf/mfu_pct": redline_metrics.get("mfu_pct", 0.0),
        "memory/max_reserved_gb": redline_metrics.get("max_reserved_gb", 0.0),
        "memory/max_reserved_pct": redline_metrics.get("max_reserved_pct", 0.0),
    }

    return metrics


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
