from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
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
    Static {
        text-align: center;
        content-align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(Text(ascii_art))

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.app.push_screen("hud")
