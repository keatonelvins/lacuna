import os
import subprocess
from pathlib import Path
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual.containers import Vertical, Center
from textual import events
from rich.text import Text

ascii_art = """
░  ░░░░░░░░░      ░░░░      ░░░  ░░░░  ░░   ░░░  ░░░      ░░
▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒    ▒▒  ▒▒  ▒▒▒▒  ▒
▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓  ▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓  ▓▓  ▓  ▓  ▓▓  ▓▓▓▓  ▓
█  ████████        ██  ████  ██  ████  ██  ██    ██        █
█        ██  ████  ███      ████      ███  ███   ██  ████  █
""".strip()

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

    OptionList {
        border: solid white;
        width: 60%;
        height: auto;
        max-height: 20;
    }

    OptionList > .option-list--option {
        color: white;
    }

    OptionList > .option-list--option-highlighted {
        background: #333333;
        color: white;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(ascii_art))
            with Center():
                yield OptionList(id="config_list")

    def on_mount(self) -> None:
        self._load_config_files()

    def _load_config_files(self) -> None:
        option_list = self.query_one("#config_list", OptionList)
        configs_dir = Path("configs")

        if configs_dir.exists() and configs_dir.is_dir():
            toml_files = list(configs_dir.glob("*.toml"))
            if toml_files:
                for toml_file in sorted(toml_files):
                    display_name = toml_file.stem
                    option_list.add_option(Option(display_name, id=str(toml_file)))

        option_list.add_option(Option("continue...", id="__continue__"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.disabled:
            return

        config_path = event.option.id
        if config_path == "__continue__":
            self.app.push_screen("hud")
        elif config_path:
            self._run_training(config_path)

    def _run_training(self, config_path: str) -> None:
        try:
            subprocess.Popen([
                "tmux", "new-session", "-d", "-s", f"lacuna_train_{os.getpid()}",
                "uv", "run", "train", config_path
            ])
            self.app.push_screen("hud")
        except Exception as e:
            pass

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            option_list = self.query_one("#config_list", OptionList)
            if option_list.highlighted is not None:
                option = option_list.get_option_at_index(option_list.highlighted)
                if option and not option.disabled and option.id:
                    if option.id == "__continue__":
                        self.app.push_screen("hud")
                    else:
                        self._run_training(option.id)
