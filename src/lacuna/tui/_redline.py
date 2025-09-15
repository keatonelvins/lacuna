from textual.widgets import Static
from .utils import get_latest_metrics


class RedlineWidget(Static):
    def __init__(self) -> None:
        super().__init__()
        self.border_title = "_redline"
        self.update_content()

    def on_mount(self) -> None:
        self.set_interval(1, self.update_content)

    def update_content(self) -> None:
        latest = get_latest_metrics()
        if latest:
            content = f"MFU: {latest['mfu_pct']:.1f}%\n"
            content += f"TFLOPS: {latest['tflops']:.1f}\n"
            content += f"Tok/s: {latest['tps']:.0f}\n"
            content += f"Memory: {latest['max_reserved_gb']:.1f}GB"
            self.update(content)
        else:
            self.update("No active run")
