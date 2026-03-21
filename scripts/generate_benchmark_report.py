from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_quick import run_benchmark


def render_markdown(timestamp: str, results: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Benchmark Report",
        "",
        f"Generated (UTC): {timestamp}",
        "",
        "| Component | avg (ms) | p50 (ms) | p90 (ms) | 95% CI (ms) |",
        "|---|---:|---:|---:|---:|",
    ]

    for component, stats in results.items():
        row = (
            f"| {component} | {stats['avg_ms']:.3f} | "
            f"{stats['p50_ms']:.3f} | {stats['p90_ms']:.3f} | "
            f"[{stats['ci95_low_ms']:.3f}, {stats['ci95_high_ms']:.3f}] |"
        )
        lines.append(
            row
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- Results are hardware-dependent and intended as reproducibility anchors.",
            "- Rerun using `python scripts/generate_benchmark_report.py`.",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    results = run_benchmark()
    timestamp = datetime.now(timezone.utc).isoformat()

    benchmarks_dir = ROOT / "experiments" / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": timestamp,
        "results": results,
    }

    json_path = benchmarks_dir / "latest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = render_markdown(timestamp, results)
    report_path = ROOT / "docs" / "BENCHMARK_REPORT.md"
    report_path.write_text(markdown + "\n", encoding="utf-8")

    print(f"Wrote JSON benchmark artifact: {json_path}")
    print(f"Wrote markdown benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())