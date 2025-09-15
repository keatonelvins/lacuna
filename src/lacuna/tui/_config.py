import json
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Static


class ConfigScreen(ModalScreen):
    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("c", "dismiss", "Close"),
    ]

    CSS = """
    ConfigScreen {
        background: $panel-lighten-1 80%;
    }

    Grid {
        grid-size: 3;
        grid-gutter: 1;
        height: auto;
        padding: 2;
    }

    .config-section {
        border: solid;
        padding: 1;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        config = self._load_config()
        sections = []

        for section_name, section_data in config.items():
            content = f"[bold]{section_name.upper()}[/bold]\n\n"
            for key, value in section_data.items():
                content += f"{key}: {value}\n"
            sections.append(Static(content.strip(), classes="config-section"))

        yield Grid(*sections)

    def _load_config(self) -> dict:
        cache_dir = Path.cwd() / ".lacuna_cache" / "active_run"
        settings_file = cache_dir / "settings.json"

        try:
            with open(settings_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": {"message": "Could not load settings.json"}}