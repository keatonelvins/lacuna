from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Tree, Static

from lacuna.tui.utils import get_settings


class ConfigScreen(ModalScreen):
    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("c", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        config = get_settings()

        if config is None:
            yield Static("No active run")
            return

        tree: Tree[str] = Tree("config")
        tree.root.expand()

        for section_name, section_data in config.items():
            section_node = tree.root.add(f"{section_name}", expand=True)
            for key, value in section_data.items():
                section_node.add_leaf(f"[dim][/][bold cyan]{key}[/]: [white]{value}[/]")

        yield tree
