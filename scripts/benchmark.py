"""
Benchmark tps, mfu, time, and memory for all configs in a given directory.

Usage: 
uv run scripts/benchmark.py path/to/configs/
"""

import sys
import json
import subprocess
from pathlib import Path


def get_last_metrics(run_dir, n=10):
    """Average metrics over last n lines of metrics.jsonl"""
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None

    lines = []
    with open(metrics_file) as f:
        for line in f:
            lines.append(json.loads(line))

    if not lines:
        return None

    recent = lines[-n:]
    keys = ["perf/throughput(tps)", "perf/mfu(%)", "time_metrics/end_to_end(s)", "memory/max_reserved(GiB)"]
    return {k: sum(m.get(k, 0) for m in recent) / len(recent) for k in keys}


def run_benchmark(config_dir: str = "configs"):
    """Run training on all configs and report metrics"""
    configs = sorted(Path(config_dir).glob("*.toml"))
    results = []

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Running: {config.name}")
        print("=" * 60)

        subprocess.run(["uv", "run", "train", str(config)], check=True)

        cache_dir = Path(".lacuna_cache/runs")
        latest_run = max(cache_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        metrics = get_last_metrics(latest_run)

        if metrics:
            results.append(
                {
                    "config": config.name,
                    "throughput(tps)": metrics.get("perf/throughput(tps)", 0),
                    "mfu(%)": metrics.get("perf/mfu(%)", 0),
                    "end_to_end(s)": metrics.get("time_metrics/end_to_end(s)", 0),
                    "memory(GiB)": metrics.get("memory/max_reserved(GiB)", 0),
                }
            )

    print(f"\n\n{'=' * 80}")
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Config':<30} {'TPS':>12} {'MFU%':>8} {'Time(s)':>10} {'Memory(GiB)':>12}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['config']:<30} {r['throughput(tps)']:>12.1f} {r['mfu(%)']:>8.2f} {r['end_to_end(s)']:>10.3f} {r['memory(GiB)']:>12.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    config_dir = sys.argv[1] if len(sys.argv) > 1 else "configs"
    run_benchmark(config_dir)
