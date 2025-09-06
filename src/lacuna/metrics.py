"""MFU, memory, and throughput tracking."""

import time
from collections import deque
import torch
from loguru import logger
from pydantic import BaseModel


class StateTracker(BaseModel):
    step: int = 0
    total_tokens: int = 0
    peak_mfu: float = 0.0
    peak_tflops: float = 0.0
    peak_mem_gb: float = 0.0


# ref: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
def get_peak_flops(device_name: str) -> int:
    if "H100" in device_name:
        return 989e12  # assuming H100 SXM bc otherwise what are you doing my guy
    elif "H200" in device_name:
        return 989e12
    elif "B200" in device_name:
        return 2.25e15
    elif "3090" in device_name:
        return 142e12
    else:
        logger.warning(f"Peak flops undefined for: {device_name}, falling back to A100")
        return 312e12


def calculate_model_flops(model: torch.nn.Module, seq_len: int) -> tuple[int, int]:
    """Get parameter count and FLOPs/token at seq_len.

    TODO: this is not super exact, can improve later
    """
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
        window_sec: float = 30.0,
        device: torch.device | None = None,
    ):
        self.seq_len = seq_len
        self.world_size = world_size
        self.window_sec = window_sec
        self.device = device or torch.device("cuda", torch.cuda.current_device())

        self._tokens: deque[tuple[float, int]] = deque()
        self._steps: deque[tuple[float, float]] = deque()  # (timestamp, step_time)
        self._data_load_times: deque[tuple[float, float]] = deque()  # (timestamp, load_time)

        self.device_name = torch.cuda.get_device_name(self.device)
        self.gpu_peak_flops = get_peak_flops(self.device_name)
        self.num_params, self.flops_per_token = calculate_model_flops(model, seq_len)

        self.device_props = torch.cuda.get_device_properties(self.device)
        self.total_memory = self.device_props.total_memory

        self._last_time: float | None = None
        self._start_time = time.perf_counter()

        self.state = StateTracker()

    def update(self, tokens: int, data_load_time: float | None = None) -> None:
        """Update tracker with tokens processed."""
        now = time.perf_counter()
        self._tokens.append((now, tokens))

        self.state.total_tokens += tokens
        self.state.step += 1

        if self._last_time is not None:
            step_time = now - self._last_time
            self._steps.append((now, step_time))
        self._last_time = now

        if data_load_time is not None:
            self._data_load_times.append((now, data_load_time))

        self._shed(now)

    def read(self) -> dict[str, float]:
        """Get all metrics as a dict."""
        now = time.perf_counter()
        self._shed(now)

        metrics = {}

        tps = self._get_tps(now)
        if tps is not None:
            metrics["tps"] = tps

            actual_flops = self.flops_per_token * tps
            metrics["mfu_pct"] = 100 * actual_flops / self.gpu_peak_flops
            metrics["tflops"] = actual_flops / 1e12

            self.state.peak_mfu = max(self.state.peak_mfu, metrics["mfu_pct"])
            self.state.peak_tflops = max(self.state.peak_tflops, metrics["tflops"])

        if self._steps:
            avg_step_time = sum(t for _, t in self._steps) / len(self._steps)
            metrics["latency_ms"] = avg_step_time * 1000
            metrics["steps_per_s"] = 1.0 / avg_step_time if avg_step_time > 0 else 0.0

        if self._data_load_times:
            total_data_time = sum(t for _, t in self._data_load_times)
            total_time = now - self._start_time
            metrics["data_pct"] = 100 * total_data_time / max(total_time, 0.001)
        else:
            metrics["data_pct"] = 0.0

        metrics.update(self._get_memory())

        self.state.peak_mem_gb = max(self.state.peak_mem_gb, metrics.get("max_reserved_gb", 0.0))

        return metrics

    def clear_data_times(self) -> None:
        """Clear data loading times after logging."""
        self._data_load_times.clear()

    def _shed(self, now: float) -> None:
        """Remove data outside the time window."""
        cutoff = now - self.window_sec

        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()
        while self._steps and self._steps[0][0] < cutoff:
            self._steps.popleft()
        while self._data_load_times and self._data_load_times[0][0] < cutoff:
            self._data_load_times.popleft()

    def _get_tps(self, now: float) -> float | None:
        """Calculate tokens per second per device."""
        if len(self._tokens) < 2:
            return None

        total_tokens = sum(tokens for _, tokens in self._tokens)
        time_span = max(1e-9, now - self._tokens[0][0])

        return total_tokens / time_span / self.world_size

    def _get_memory(self) -> dict[str, float]:
        """Get memory statistics in GB and percentage."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {"max_reserved_gb": 0.0, "max_reserved_pct": 0.0}

        gb = 1024**3
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        max_reserved_gb = max_reserved / gb
        max_reserved_pct = 100 * max_reserved / self.total_memory if self.total_memory > 0 else 0.0

        return {
            "max_reserved_gb": max_reserved_gb,
            "max_reserved_pct": max_reserved_pct,
        }
