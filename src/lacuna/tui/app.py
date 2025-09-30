from textual.app import App
from textual.binding import Binding

from ._config import ConfigScreen
from ._logs import LogScreen
from ._menu import MenuScreen
from ._hud import HudScreen


class Lacuna(App):
    SCREENS = {
        "menu": MenuScreen,
        "hud": HudScreen,
        "config": ConfigScreen,
        "logs": LogScreen,
    }

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        """Start with the menu screen."""
        self.push_screen("menu")


def main():
    app = Lacuna()
    app.run()


if __name__ == "__main__":
    main()
