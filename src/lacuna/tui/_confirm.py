import os
import subprocess
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.containers import Vertical


class ConfirmScreen(ModalScreen):
    CSS = """
    ConfirmScreen {
        align: center middle;
    }

    Vertical {
        width: 80%;
        height: auto;
        border: solid white;
        padding: 1;
        background: $background;
    }

    .title {
        text-align: center;
        text-style: bold;
        color: white;
        margin-bottom: 1;
    }

    .command {
        text-align: center;
        color: cyan;
        margin-bottom: 2;
    }

    .hint {
        text-align: center;
        text-style: dim;
    }
    """

    BINDINGS = [
        ("enter", "confirm", "Confirm"),
        ("escape", "cancel", "Cancel"),
        ("ctrl+c", "exit", "Exit"),
    ]

    def __init__(self, command: str):
        super().__init__()
        self.command = command

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Run Command:", classes="title")
            yield Static(self.command, classes="command")
            yield Static("press enter to confirm, escape to cancel", classes="hint")

    def action_confirm(self) -> None:
        parts = self.command.split()
        subprocess.Popen(["tmux", "new-session", "-d", "-s", f"lacuna_train_{os.getpid()}", "uv", "run", "train"] + parts)
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_exit(self) -> None:
        self.app.exit()
