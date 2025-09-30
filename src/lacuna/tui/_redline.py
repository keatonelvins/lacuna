from textual.widgets import Static
from rich.table import Table
from rich.panel import Panel
from rich import box

from .utils import get_gpu_hardware, get_latest_metrics


class RedlineWidget(Static):
    def __init__(self) -> None:
        super().__init__()
        self.gpu_hw_data = []
        self.training_metrics = None

    def on_mount(self) -> None:
        self.set_interval(1.0, self.update_content)
        self.update_content()

    def _render_gpu_card(self, gpu_idx: int) -> str:
        gpu_data = self.gpu_hw_data[gpu_idx] if gpu_idx < len(self.gpu_hw_data) else None

        if not gpu_data:
            return f"GPU{gpu_idx} ---"

        return f"GPU{gpu_idx} {gpu_data['gpu_util']}% {gpu_data['temp']}° {gpu_data['mem_used_gb']:.1f}/{gpu_data['mem_total_gb']:.0f}GB"

    def _render_perf_panel(self) -> Table:
        panel_table = Table.grid(padding=(0, 0))
        panel_table.add_column()

        if not self.training_metrics:
            panel_table.add_row("No metrics")
            return panel_table

        metrics = self.training_metrics

        mfu_val = metrics["mfu"]
        mfu_box = Panel(f"{mfu_val:.1f}%", title="MFU", box=box.SQUARE)
        panel_table.add_row(mfu_box)

        tok_s_val = metrics["throughput"]
        tok_s = f"{tok_s_val / 1000:.1f}k" if tok_s_val > 1000 else f"{tok_s_val:.0f}"
        panel_table.add_row(f"tok/s {tok_s}")
        panel_table.add_row(f"tflops {metrics['tflops']:.1f}")
        panel_table.add_row(f"mem {metrics['max_active_pct']:.0f}%")
        panel_table.add_row(f"io {metrics['data_loading_pct']:.1f}%")
        panel_table.add_row(f"oom {metrics['num_ooms']} retries {metrics['num_alloc_retries']}")

        return panel_table

    def render(self) -> Table:
        """Render the complete redline widget layout."""
        main_table = Table.grid(padding=(0, 2))
        main_table.add_column()
        main_table.add_column()

        gpu_lines = []
        for i in range(8):
            gpu_lines.append(self._render_gpu_card(i))

        main_table.add_row("\n".join(gpu_lines), self._render_perf_panel())
        return main_table

    def update_content(self) -> None:
        """Fetch data and update display."""
        self.gpu_hw_data = get_gpu_hardware()
        self.training_metrics = get_latest_metrics()
        self.update(self.render())
