from pathlib import Path
from pydantic import BaseModel
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Input
from textual.containers import Vertical, Center
from rich.text import Text
from textual_autocomplete import AutoComplete, DropdownItem
from lacuna.config import LacunaConfig

ascii_art = """
░  ░░░░░░░░░      ░░░░      ░░░  ░░░░  ░░   ░░░  ░░░      ░░
▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒    ▒▒  ▒▒  ▒▒▒▒  ▒
▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓  ▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓  ▓▓  ▓  ▓  ▓▓  ▓▓▓▓  ▓
█  ████████        ██  ████  ██  ████  ██  ██    ██        █
█        ██  ████  ███      ████      ███  ███   ██  ████  █
""".strip()

def _flatten_pydantic_fields(model: type[BaseModel], prefix: str = "") -> list[str]:
    fields = []
    for name, field_info in model.model_fields.items():
        field_name = f"{prefix}.{name}" if prefix else name
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            fields.extend(_flatten_pydantic_fields(annotation, field_name))
        else:
            fields.append(f"--{field_name}")
    return fields

class MenuScreen(Screen):
    CSS = """
    Vertical {
        height: 100%;
        align: center middle;
    }

    Static {
        text-align: center;
        margin-bottom: 2;
    }

    Input {
        border: solid white;
        width: 60%;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(ascii_art))
            with Center():
                input_widget = Input(placeholder="configs/model.toml --flag.value (or enter to skip)", id="train_input")
                yield input_widget
                yield AutoComplete(input_widget, candidates=self._get_candidates())

    def _get_candidates(self) -> list[DropdownItem]:
        candidates = []
        configs_dir = Path("configs")
        if configs_dir.exists() and configs_dir.is_dir():
            candidates.extend([DropdownItem(str(f)) for f in sorted(configs_dir.glob("*.toml"))])
        candidates.extend([DropdownItem(flag) for flag in _flatten_pydantic_fields(LacunaConfig)])
        return candidates

    def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        if not command:
            self.app.push_screen("hud")
        else:
            from lacuna.tui._confirm import ConfirmScreen
            self.app.push_screen(ConfirmScreen(command), self._on_confirm)

    def _on_confirm(self, confirmed: bool) -> None:
        if confirmed:
            self.app.push_screen("hud")
