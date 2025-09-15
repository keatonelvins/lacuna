import json
from pathlib import Path
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Tree


class ConfigScreen(ModalScreen):
    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("c", "dismiss", "Close"),
    ]

    CSS = """
    ConfigScreen {
        background: $panel-darken-2 90%;
        border: heavy $accent;
    }

    Tree {
        background: $panel-darken-1;
        border: round $primary;
        height: 1fr;
        margin: 1;
        padding: 1;
    }

    Tree:focus {
        border: heavy $accent;
        background: $panel-darken-1 90%;
    }

    TreeNode {
        color: $text;
    }

    TreeNode.-expanded {
        color: $accent;
        text-style: bold;
    }

    TreeNode.-highlighted {
        background: $accent 20%;
        color: $accent;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        config = self._load_config()
        tree: Tree[str] = Tree("config")
        tree.root.expand()

        for section_name, section_data in config.items():
            section_node = tree.root.add(f"{section_name}", expand=True)
            for key, value in section_data.items():
                section_node.add_leaf(f"[dim]•[/] [bold cyan]{key}[/]: [white]{value}[/]")

        yield tree

    def _load_config(self) -> dict:
        cache_dir = Path.cwd() / ".lacuna_cache" / "active_run"
        settings_file = cache_dir / "settings.json"

        try:
            with open(settings_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": {"message": "Could not load settings.json"}}