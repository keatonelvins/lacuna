from textual.widgets import Static
from .utils import get_gpu_hardware


class RedlineWidget(Static):
    def __init__(self) -> None:
        super().__init__()
        self.update_content()

    def on_mount(self) -> None:
        self.set_interval(1, self.update_content)

    def update_content(self) -> None:
        gpu_hw = get_gpu_hardware()

        content_lines = []

        for i, gpu in enumerate(gpu_hw):
            content_lines.append(
                f"GPU{i}: {gpu['gpu_util']}% util | {gpu['temp']}°C | {gpu['power']}W | {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.1f}GB"
            )

        self.update("\n".join(content_lines))
