from textual.widgets import Static
from .utils import get_latest_metrics


class CurvesWidget(Static):
    def __init__(self) -> None:
        super().__init__()
        self.border_title = "_curves"
        self.update_content()

    def on_mount(self) -> None:
        self.set_interval(1, self.update_content)

    def update_content(self) -> None:
        latest = get_latest_metrics()
        if latest:
            content = f"Loss: {latest['loss']:.4f}\n"
            content += f"LR: {latest['lr']:.2e}\n"
            content += f"Grad Norm: {latest['grad_norm']:.2f}"
            self.update(content)
        else:
            self.update("No active run")
