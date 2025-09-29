import os
import subprocess
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Tree, Static
from textual.containers import Vertical
from rich.text import Text
from lacuna.config import LacunaConfig

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

    Static {
        text-align: center;
        margin-bottom: 1;
    }

    Tree {
        height: auto;
        max-height: 30;
        scrollbar-size: 1 1;
    }
    """

    BINDINGS = [
        ("enter", "confirm", "Confirm"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, command: str):
        super().__init__()
        self.command = command

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(f"[bold white]Run Command:[/]\n[cyan]{self.command}[/]", justify="center"))
            yield Static(Text("[dim]press enter to confirm, escape to cancel[/]", justify="center"))
            tree = self._build_config_tree()
            yield tree

    def _build_config_tree(self) -> Tree:
        parts = self.command.split()
        config_args = ["--cli_parse_args=true"] + parts
        config = LacunaConfig(_cli_parse_args=config_args)

        tree: Tree[str] = Tree("configuration")
        tree.root.expand()

        for section_name, section_data in config.model_dump().items():
            section_node = tree.root.add(f"[bold]{section_name}[/]", expand=False)
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    section_node.add_leaf(f"[cyan]{key}[/]: [white]{value}[/]")
            else:
                section_node.add_leaf(f"[white]{section_data}[/]")

        return tree

    def action_confirm(self) -> None:
        parts = self.command.split()
        subprocess.Popen(["tmux", "new-session", "-d", "-s", f"lacuna_train_{os.getpid()}", "uv", "run", "train"] + parts)
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)