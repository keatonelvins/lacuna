from textual.widgets import Static
from .utils import get_latest_metrics, get_gpu_hardware


class RedlineWidget(Static):
    def __init__(self) -> None:
        super().__init__()
        self.border_title = "_redline"
        self.update_content()

    def on_mount(self) -> None:
        self.set_interval(1, self.update_content)

    def update_content(self) -> None:
        latest = get_latest_metrics()
        gpu_hw = get_gpu_hardware()

        content_lines = []

        if latest:
            content_lines.extend([
                f"MFU: {latest['mfu_pct']:.1f}%",
                f"TFLOPS: {latest['tflops']:.1f}",
                f"Tok/s: {latest['tps']:.0f}",
                f"Memory: {latest['max_reserved_gb']:.1f}GB"
            ])
        else:
            content_lines.append("No active run")

        for i, gpu in enumerate(gpu_hw):
            content_lines.extend([
                f"GPU{i}: {gpu['gpu_util']}% util",
                f"Temp: {gpu['temp']}°C",
                f"Power: {gpu['power']}W",
                f"VRAM: {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.1f}GB"
            ])

        if not gpu_hw:
            content_lines.append("No GPUs detected")

        self.update("\n".join(content_lines))
