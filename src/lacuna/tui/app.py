# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "textual-dev",
# ]
# ///

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid

from _curves import CurvesWidget
from _data import DataWidget
from _i_o import IOWidget
from _redline import RedlineWidget
from _splash import SplashScreen


class Lacuna(App):
    BINDINGS = [Binding("ctrl+c", "quit", "Quit")]

    CSS = """
    Grid {
        grid-size: 2;
        grid-columns: 2fr 1fr;
        grid-rows: 2fr 1fr;
        grid-gutter: 1;
        height: 100%;
    }

    RedlineWidget {
        border: solid red;
        height: 100%;
    }

    DataWidget {
        border: solid green;
        height: 100%;
    }

    CurvesWidget {
        border: solid cornflowerblue;
        height: 100%;
    }

    IOWidget {
        border: solid yellow;
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Grid(
            RedlineWidget(),
            DataWidget(),
            CurvesWidget(),
            IOWidget(),
        )

    def on_mount(self) -> None:
        """Show splash screen on app startup."""
        self.push_screen(SplashScreen())


def main():
    app = Lacuna()
    app.run()


if __name__ == "__main__":
    main()
