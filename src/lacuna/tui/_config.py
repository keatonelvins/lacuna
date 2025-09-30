from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Tree, Static
from textual.containers import Container

from lacuna.tui.utils import get_settings


class ConfigScreen(ModalScreen):
    CSS = """
    ConfigScreen {
        align: center middle;
        background: $background 30%;
    }

    Container {
        width: 90%;
        height: 90%;
        border: solid $accent;
    }

    Tree {
        width: 100%;
        height: 100%;
    }

    Static {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", priority=True),
        Binding("c", "dismiss", "Close", priority=True),
        Binding("ctrl+c", "dismiss", "Close", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Container()

    def on_mount(self) -> None:
        """Load config when screen is first mounted."""
        self.load_config()

    def on_screen_resume(self) -> None:
        """Reload config when screen is resumed."""
        self.load_config()

    def load_config(self) -> None:
        """Load and display the config."""
        container = self.query_one(Container)
        container.remove_children()

        config = get_settings()

        if config is None:
            container.mount(Static("No active run"))
            return

        tree: Tree[str] = Tree("config")
        tree.root.expand()

        for section_name, section_data in config.items():
            section_node = tree.root.add(f"{section_name}", expand=True)
            for key, value in section_data.items():
                section_node.add_leaf(f"[dim][/][bold cyan]{key}[/]: [white]{value}[/]")

        container.mount(tree)
