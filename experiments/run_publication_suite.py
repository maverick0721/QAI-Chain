from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_research_suite.py"),
        "--episodes",
        "45",
        "--steps",
        "30",
        "--seeds",
        "3,7,11,17,21,42,84,126,5,9,13,19,23,31,57,99",
        "--out-json",
        "experiments/results/research_results.json",
        "--out-md",
        "docs/RESEARCH_RESULTS.md",
    ]

    print("Running publication-scale suite...")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())