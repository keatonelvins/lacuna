from pathlib import Path
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Log


class LogScreen(ModalScreen):
    """Modal screen for displaying training logs."""

    CSS = """
    LogScreen {
        align: center middle;
        background: $background 30%;
    }

    Log {
        width: 90%;
        height: 90%;
        border: solid $accent;
        scrollbar-background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", priority=True),
        Binding("l", "dismiss", "Close", priority=True),
        Binding("ctrl+c", "dismiss", "Close", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Log(auto_scroll=True)

    def on_mount(self) -> None:
        """Load logs when screen is mounted."""
        self.set_interval(0.5, self.update_logs)
        self.last_line_count = 0
        self.update_logs()

    def update_logs(self) -> None:
        """Read and display logs from the active run."""
        log_file = Path(".lacuna_cache/active_run/run.log")

        if not log_file.exists():
            log = self.query_one(Log)
            if self.last_line_count == 0:
                log.clear()
                log.write_line("No active training run")
                self.last_line_count = -1
            return

        try:
            with open(log_file) as f:
                lines = f.readlines()

            if len(lines) > self.last_line_count:
                log = self.query_one(Log)

                if self.last_line_count <= 0:
                    log.clear()

                # Write new lines
                for line in lines[max(0, self.last_line_count) :]:
                    log.write_line(line.rstrip())

                self.last_line_count = len(lines)

        except Exception as e:
            log = self.query_one(Log)
            log.write_line(f"Error reading log: {e}")
