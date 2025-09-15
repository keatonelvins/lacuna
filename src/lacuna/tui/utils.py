import json
from pathlib import Path
import pynvml


def get_latest_metrics():
    """Get the latest metrics from the active run."""
    metrics_file = Path(".lacuna_cache/active_run/metrics.jsonl")
    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1].strip())


def get_settings():
    """Get settings from the active run."""
    settings_file = Path(".lacuna_cache/active_run/settings.json")
    if not settings_file.exists():
        return None

    with open(settings_file) as f:
        return json.load(f)


def get_gpu_hardware():
    """Get GPU hardware metrics using pynvml."""
    try:
        pynvml.nvmlInit()
    except:
        return []

    device_count = pynvml.nvmlDeviceGetCount()

    gpus = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        mem_util = util.memory

        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_gb = mem_info.used / (1024**3)
        mem_total_gb = mem_info.total / (1024**3)

        gpus.append({
            'gpu_util': gpu_util,
            'mem_util': mem_util,
            'temp': temp,
            'power': power,
            'mem_used_gb': mem_used_gb,
            'mem_total_gb': mem_total_gb
        })

    pynvml.nvmlShutdown()
    return gpus