from textual.screen import Screen
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid

from ._curves import CurvesWidget
from ._i_o import IOWidget
from ._redline import RedlineWidget


class HudScreen(Screen):
    CSS = """
    Grid {
        grid-size: 2;
        grid-columns: 2fr 1fr;
        grid-rows: 2fr 1fr;
        grid-gutter: 0 1;
        height: 100%;
    }

    RedlineWidget {
        border: solid white;
        column-span: 2;
        height: 100%;
    }

    CurvesWidget {
        border: solid white;
        height: 100%;
    }

    IOWidget {
        border: solid white;
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("c", "config", "Config"),
        Binding("l", "logs", "Logs"),
        Binding("escape", "menu", "Menu"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Grid(
            RedlineWidget(),
            CurvesWidget(),
            IOWidget(),
        )

    def action_config(self) -> None:
        self.app.push_screen("config")

    def action_logs(self) -> None:
        self.app.push_screen("logs")

    def action_menu(self) -> None:
        self.app.pop_screen()
