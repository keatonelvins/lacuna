from pathlib import Path
from pydantic import BaseModel
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Input
from textual.containers import Vertical, Center
from rich.text import Text
from textual_autocomplete import AutoComplete, DropdownItem
from textual_autocomplete._autocomplete import TargetState

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
        if isinstance(field_info.annotation, type) and issubclass(field_info.annotation, BaseModel):
            fields.extend(_flatten_pydantic_fields(field_info.annotation, field_name))
        else:
            fields.append(f"--{field_name}")
    return fields


_CONFIG_FLAGS = None


class ConfigAutoComplete(AutoComplete):
    def _get_flags(self):
        global _CONFIG_FLAGS
        if _CONFIG_FLAGS is None:
            from lacuna.config import LacunaConfig

            _CONFIG_FLAGS = [DropdownItem(f) for f in _flatten_pydantic_fields(LacunaConfig)]
        return _CONFIG_FLAGS

    def get_candidates(self, state: TargetState) -> list[DropdownItem]:
        has_toml = any(w.endswith(".toml") for w in state.text.split())
        candidates = []
        if not has_toml:
            configs_dir = Path("configs")
            if configs_dir.is_dir():
                candidates.extend([DropdownItem(str(f)) for f in sorted(configs_dir.glob("*.toml"))])
        candidates.extend(self._get_flags())
        return candidates

    def get_search_string(self, state: TargetState) -> str:
        words = state.text[: state.cursor_position].split()
        return words[-1] if words else ""

    def apply_completion(self, value: str, state: TargetState) -> None:
        if not self.target:
            return
        before = state.text[: state.cursor_position].rsplit(" ", 1)
        prefix = before[0] + " " if len(before) > 1 else ""
        self.target.value = prefix + value + " " + state.text[state.cursor_position :]
        self.target.cursor_position = len(prefix) + len(value) + 1


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
                yield ConfigAutoComplete(input_widget, candidates=None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            self.app.push_screen("hud")
        else:
            from lacuna.tui._confirm import ConfirmScreen

            self.app.push_screen(ConfirmScreen(event.value.strip()), lambda c: c and self.app.push_screen("hud"))

    def on_key(self, event) -> None:
        if event.key == "ctrl+c":
            self.app.exit()
