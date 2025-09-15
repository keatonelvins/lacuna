import json
from pathlib import Path


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