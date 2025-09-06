import json
from pathlib import Path
from loguru import logger
from rich.pretty import Pretty
from rich.console import Console

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
        json.dump(state.model_dump(), f, indent=4)


def save_settings_json(path: Path, config: LacunaConfig) -> None:
    if not is_master():
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / "settings.json").open("w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=4)


def load_state_json(path: Path) -> StateTracker:
    ts_path = path / "state.json"
    if ts_path.exists():
        with ts_path.open("r") as f:
            return StateTracker(**json.load(f))
    return StateTracker()
