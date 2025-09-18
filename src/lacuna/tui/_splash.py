from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static


class SplashScreen(ModalScreen):
    class DismissScreen(Message):
        """Message to dismiss the splash screen."""

    CSS = """
    SplashScreen {
        background: $background;
        align: center middle;
    }

    #splash-container {
        width: auto;
        height: auto;
        padding: 2 4;
        align: center middle;
    }

    #ascii-art {
        text-align: center;
        width: auto;
        height: auto;
    }

    #loading-text {
        text-align: center;
        margin-top: 2;
        color: $text-muted;
        width: auto;
    }
    """

    def __init__(self):
        super().__init__()

    def _create_logo(self) -> Text:
        """Create static ASCII art logo."""
        logo = Text()
        logo.append("╔═══════════════════════════════════════════════════════╗\n", style="cyan")
        logo.append("║                                                       ║\n", style="cyan")
        logo.append("║  ██╗      █████╗  ██████╗██╗   ██╗███╗   ██╗ █████╗   ║\n", style="cyan bold")
        logo.append("║  ██║     ██╔══██╗██╔════╝██║   ██║████╗  ██║██╔══██╗  ║\n", style="cyan bold")
        logo.append("║  ██║     ███████║██║     ██║   ██║██╔██╗ ██║███████║  ║\n", style="cyan bold")
        logo.append("║  ██║     ██╔══██║██║     ██║   ██║██║╚██╗██║██╔══██║  ║\n", style="cyan bold")
        logo.append("║  ███████╗██║  ██║╚██████╗╚██████╔╝██║ ╚████║██║  ██║  ║\n", style="cyan bold")
        logo.append("║  ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝  ║\n", style="cyan bold")
        logo.append("║                                                       ║\n", style="cyan")
        logo.append("╚═══════════════════════════════════════════════════════╝", style="cyan")
        return logo

    def compose(self) -> ComposeResult:
        with Middle():
            with Center(id="splash-container"):
                yield Static(self._create_logo(), id="ascii-art")

    def on_mount(self) -> None:
        """Auto-dismiss after 2 seconds."""
        self.set_timer(1.0, lambda: self.post_message(self.DismissScreen()))

    def on_splash_screen_dismiss_screen(self) -> None:
        """Handle the dismiss message."""
        self.dismiss()
