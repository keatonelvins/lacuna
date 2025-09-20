"""MFU, memory, and throughput tracking."""

import time
import torch
from collections import deque
from loguru import logger


# ref: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
def get_peak_flops(device_name: str) -> int:
    if "H100" or "H200" in device_name:
        return 989e12  # assuming H100 SXM bc otherwise what are you doing my guy
    elif "B200" in device_name:
        return 2.25e15
    elif "3090" in device_name:
        return 142e12
    else:
        logger.warning(f"Peak flops undefined for: {device_name}, falling back to A100")
        return 312e12


def calculate_model_flops(model: torch.nn.Module, seq_len: int) -> tuple[int, int]:
    """Get parameter count and FLOPs/token at seq_len."""
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    head_dim = config.hidden_size // config.num_attention_heads

    attn_flops = 12 * config.num_hidden_layers * config.num_attention_heads * head_dim * seq_len
    flops_per_token = 6 * num_params + attn_flops

    return int(num_params), int(flops_per_token)


class Redline:
    """GPU performance metrics tracker."""

    def __init__(
        self,
        model: torch.nn.Module,
        seq_len: int,
        world_size: int = 1,
        window_steps: int = 100,
        start_step: int = 0,
    ):
        self.seq_len = seq_len
        self.world_size = world_size
        self.window_steps = window_steps
        self.device = torch.device("cuda")

        self._tokens: deque[int] = deque(maxlen=window_steps)
        self._step_times: deque[float] = deque(maxlen=window_steps)
        self._data_load_times: deque[float] = deque(maxlen=window_steps)
        self._last_time: float | None = None

        self.gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(self.device))
        self.num_params, self.flops_per_token = calculate_model_flops(model, seq_len)
        self.device_props = torch.cuda.get_device_properties(self.device)
        self.total_memory = self.device_props.total_memory

        self.step = start_step
        self.total_tokens = 0

        torch.cuda.reset_peak_memory_stats()

    def update(self, tokens: int, data_load_time: float) -> None:
        """Update tracker with tokens processed."""
        now = time.perf_counter()
        self._tokens.append(tokens)
        self._data_load_times.append(data_load_time)

        self.total_tokens += tokens
        self.step += 1

        if self._last_time is not None:
            step_time = now - self._last_time
            self._step_times.append(step_time)
        self._last_time = now

    def read(self) -> dict[str, float]:
        """Get all metrics as a dict."""
        metrics = {
            "tps": 0.0,
            "mfu_pct": 0.0,
            "tflops": 0.0,
            "data_pct": 0.0,
            "max_reserved_gb": 0.0,
            "max_reserved_pct": 0.0,
        }

        if not self._step_times:
            return metrics

        total_step_time = sum(self._step_times)
        total_data_time = sum(self._data_load_times)

        tps = self._get_tps()
        metrics["tps"] = tps

        actual_flops = self.flops_per_token * tps
        metrics["mfu_pct"] = 100 * actual_flops / self.gpu_peak_flops
        metrics["tflops"] = actual_flops / 1e12
        metrics["data_pct"] = 100 * total_data_time / total_step_time

        metrics.update(self._get_memory())

        return metrics

    def _get_tps(self) -> float:
        """Calculate tokens per second per device."""
        if not self._step_times:
            return 0.0

        total_tokens = sum(self._tokens)
        total_time = sum(self._step_times)

        return total_tokens / total_time

    def _get_memory(self) -> dict[str, float]:
        """Get memory statistics in GB and percentage."""
        gb = 1024**3
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        max_reserved_gb = max_reserved / gb
        max_reserved_pct = 100 * max_reserved / self.total_memory

        return {
            "max_reserved_gb": max_reserved_gb,
            "max_reserved_pct": max_reserved_pct,
        }
