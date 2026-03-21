from __future__ import annotations

import statistics
import time
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ai.models.metrics_encoder import MetricsEncoder
from quantum.models.qnn import QNN
from quantum.transformer.q_transformer import QTransformer


def timed_forward(fn, warmup: int = 3, runs: int = 10) -> list[float]:
    for _ in range(warmup):
        fn()

    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return latencies_ms


def summarize(name: str, values: list[float]) -> None:
    p50 = statistics.median(values)
    p90 = sorted(values)[int(0.9 * (len(values) - 1))]
    avg = statistics.mean(values)
    print(f"{name:16s} avg={avg:8.3f} ms  p50={p50:8.3f} ms  p90={p90:8.3f} ms")


def summary_stats(values: list[float]) -> dict[str, float]:
    p50 = statistics.median(values)
    p90 = sorted(values)[int(0.9 * (len(values) - 1))]
    avg = statistics.mean(values)
    if len(values) > 1:
        std = statistics.stdev(values)
    else:
        std = 0.0
    half_width = 1.96 * (std / math.sqrt(len(values))) if values else 0.0
    return {
        "avg_ms": avg,
        "p50_ms": p50,
        "p90_ms": p90,
        "std_ms": std,
        "ci95_low_ms": avg - half_width,
        "ci95_high_ms": avg + half_width,
        "runs": len(values),
    }


def run_benchmark() -> dict[str, dict[str, float]]:
    torch.manual_seed(42)
    torch.set_num_threads(1)

    encoder = MetricsEncoder().eval()
    qnn = QNN().eval()
    qtransformer = QTransformer().eval()

    x_ai = torch.randn(32, 5).float()
    x_qnn = torch.randn(16, 5).float()
    x_qt = torch.randn(4, 4, 5).float()

    with torch.no_grad():
        ai_times = timed_forward(lambda: encoder(x_ai))
        qnn_times = timed_forward(lambda: qnn(x_qnn))
        qt_times = timed_forward(lambda: qtransformer(x_qt))

    return {
        "MetricsEncoder": summary_stats(ai_times),
        "QNN": summary_stats(qnn_times),
        "QTransformer": summary_stats(qt_times),
    }


def main() -> int:
    results = run_benchmark()

    print("QAI-Chain quick benchmark (CPU)")
    for name, stats in results.items():
        print(
            f"{name:16s} avg={stats['avg_ms']:8.3f} ms  "
            f"p50={stats['p50_ms']:8.3f} ms  p90={stats['p90_ms']:8.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())