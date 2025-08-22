"""MFU, memory, and throughput tracking."""

import time

import torch
from loguru import logger


def get_peak_flops(device_name: str) -> int:
    """
    Ref: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
    """
    if "H100" in device_name:
        return 989e12  # assume H100 SXM bc otherwise what are you doing my guy
    elif "H200" in device_name:
        return 989e12
    elif "B200" in device_name:
        return 2.25e15
    elif "3090" in device_name:
        return 142e12  # i use an RTX 3090 for dev
    else:
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


def calculate_model_flops(model: torch.nn.Module, seq_len: int) -> tuple[int, int]:
    """Get parameter count and FLOPs/token at seq_len.

    TODO: this is not super exact, can improve later
    """
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    head_dim = config.hidden_size // config.num_attention_heads

    attn_flops = (
        12 * config.num_hidden_layers * config.num_attention_heads * head_dim * seq_len
    )
    flops_per_token = 6 * num_params + attn_flops

    return int(num_params), int(flops_per_token)


class MFUTracker:
    """Track MFU with rolling window."""

    def __init__(
        self,
        model: torch.nn.Module,
        seq_len: int,
        window_size: int = 10,
        world_size: int = 1,
    ):
        self.window_size = window_size
        self.world_size = world_size
        self.seq_len = seq_len

        self.tokens = []
        self.times = []

        self.device_name = torch.cuda.get_device_name(torch.device("cuda"))
        self.gpu_peak_flops = get_peak_flops(self.device_name)

        self.num_params, self.flops_per_token = calculate_model_flops(model, seq_len)

        logger.info(
            f"MFU Tracker initialized: {self.device_name} "
            f"({self.gpu_peak_flops / 1e12:.1f} TFLOPS peak), "
            f"{self.num_params / 1e9:.2f}B params, "
            f"{self.flops_per_token / 1e9:.2f} GFLOPs/token"
        )

    def update(self, tokens: int) -> None:
        """Update tracker with new token count."""
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())

        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_second(self) -> float | None:
        """Get tokens per second throughput."""
        if len(self.tokens) < 2:
            return None

        total_tokens = sum(self.tokens[1:])
        time_delta = self.times[-1] - self.times[0]

        if time_delta <= 0:
            return None

        # Tokens per second per device
        return total_tokens / time_delta / self.world_size

    def get_mfu(self) -> float | None:
        """Get Model FLOPs Utilization percentage."""
        tps = self.get_tokens_per_second()
        if tps is None:
            return None

        actual_flops = self.flops_per_token * tps
        return 100 * actual_flops / self.gpu_peak_flops

    def get_tflops(self) -> float | None:
        """Get actual TFLOPS being achieved."""
        tps = self.get_tokens_per_second()
        if tps is None:
            return None

        return self.flops_per_token * tps / 1e12

    def get_metrics(self) -> dict[str, float]:
        """Get all metrics as a dict."""
        metrics = {}

        tps = self.get_tokens_per_second()
        if tps is not None:
            metrics["tokens_per_second"] = tps
            metrics["tflops"] = self.get_tflops()
            metrics["mfu_pct"] = self.get_mfu()

        return metrics


class MemoryTracker:
    """Track GPU memory usage."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda")
        self.device_props = torch.cuda.get_device_properties(self.device)
        self.total_memory = self.device_props.total_memory

        # Reset stats on init
        self.reset_peak_stats()
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics in GB."""
        stats = {}
        gb = 1024**3

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        max_allocated = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)

        # Calculate amount and percentages
        stats["allocated_gb"] = allocated / gb
        stats["allocated_pct"] = 100 * allocated / self.total_memory

        stats["reserved_gb"] = reserved / gb
        stats["reserved_pct"] = 100 * reserved / self.total_memory

        stats["max_allocated_gb"] = max_allocated / gb
        stats["max_allocated_pct"] = 100 * max_allocated / self.total_memory

        stats["max_reserved_gb"] = max_reserved / gb
        stats["max_reserved_pct"] = 100 * max_reserved / self.total_memory

        # Check for memory allocation issues
        memory_info = torch.cuda.memory_stats(self.device)
        stats["num_alloc_retries"] = memory_info.get("num_alloc_retries", 0)
        stats["num_ooms"] = memory_info.get("num_ooms", 0)

        # Warn about memory pressure
        if stats["num_alloc_retries"] > 0:
            logger.warning(
                f"GPU memory allocation retries detected: {stats['num_alloc_retries']}"
            )

        return stats

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(self.device)
