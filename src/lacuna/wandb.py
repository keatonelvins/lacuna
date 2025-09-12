from typing import Any

import wandb

from .config import LacunaConfig
from .metrics import StateTracker
from .utils import master_only


@master_only
def init_wandb(config: LacunaConfig) -> wandb.sdk.wandb_run.Run | None:
    if not config.wandb.enabled:
        return None

    run = wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        entity=config.wandb.entity,
        config=config.model_dump(mode="json"),
        mode="offline" if config.wandb.offline else "online",
    )

    return run


def log_wandb_metrics(
    loss: float,
    lr: float,
    grad_norm: float,
    state: StateTracker,
    redline_metrics: dict[str, float],
    run: wandb.sdk.wandb_run.Run | None,
) -> None:
    if run:
        metrics = {
            "train/loss": loss,
            "train/lr": lr,
            "train/grad_norm": grad_norm,
            "train/total_tokens": state.total_tokens,
            "perf/tps": redline_metrics.get("tps", 0.0),
            "perf/tflops": redline_metrics.get("tflops", 0.0),
            "perf/mfu_pct": redline_metrics.get("mfu_pct", 0.0),
            "perf/data_loading_pct": redline_metrics.get("data_pct", 0.0),
            "memory/max_reserved_gb": redline_metrics.get("max_reserved_gb", 0.0),
            "memory/max_reserved_pct": redline_metrics.get("max_reserved_pct", 0.0),
        }
        wandb.log(metrics, step=state.step)


def finish(run: wandb.sdk.wandb_run.Run | None) -> None:
    if run:
        wandb.finish()
