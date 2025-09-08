import os
import json
from pathlib import Path
from loguru import logger
from rich.pretty import Pretty
from rich.console import Console
from dataclasses import asdict

from transformers.utils.logging import disable_progress_bar

from .distributed import is_master
from .config import LacunaConfig
from .metrics import StateTracker


def setup_logger() -> None:
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        filter=lambda r: is_master(),
    )


def setup_env() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    disable_progress_bar()


def display_config(config: LacunaConfig) -> None:
    console = Console()
    with console.capture() as capture:
        console.print(Pretty(config, expand_all=True))  # omg Will you've outdone yourself
    logger.info("Starting training with config:\n" + capture.get().strip())


def save_state_json(path: Path, state: StateTracker) -> None:
    if not is_master():
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "state.json").open("w") as f:
        f.write(json.dumps(asdict(state), indent=4))


def save_settings_json(path: Path, config: LacunaConfig) -> None:
    if not is_master():
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "settings.json").open("w") as f:
        f.write(config.model_dump_json(indent=4))


def load_state_json(path: Path) -> StateTracker:
    ts_path = path / "state.json"
    if ts_path.exists():
        with ts_path.open("r") as f:
            return StateTracker(**json.load(f))
    return StateTracker()


def log_training_metrics(
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    metrics: dict[str, float],
) -> None:
    """Log training metrics in a colorful format."""
    log_parts = [
        f"\033[91mStep {step:>6}\033[0m",
        f"\033[92mLoss: {loss:7.4f}\033[0m",
        f"\033[93mGrad: {grad_norm:8.4f}\033[0m",
        f"\033[94mLR: {lr:9.2e}\033[0m",
        f"\033[36mMem: {metrics.get('max_reserved_gb', 0.0):5.1f}GB ({metrics.get('max_reserved_pct', 0.0):3.0f}%)\033[0m",
        f"\033[92mMFU: {metrics.get('mfu_pct', 0.0):5.1f}%\033[0m",
        f"\033[33mData: {metrics.get('data_pct', 0.0):5.1f}%\033[0m",
    ]

    logger.info(" | ".join(log_parts))
