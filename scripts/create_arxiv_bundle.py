from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INCLUDE_PATHS = [
    "paper",
    "docs/BENCHMARK_REPORT.md",
    "docs/RESEARCH_RESULTS.md",
    "docs/STATISTICAL_ANALYSIS.md",
    "experiments/results/research_results.json",
    "experiments/results/statistical_analysis.json",
    "experiments/benchmarks/latest.json",
]

EXCLUDE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".toc",
}


def iter_included_files() -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE_PATHS:
        p = ROOT / rel
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
            continue
        for item in p.rglob("*"):
            if item.is_file() and item.suffix not in EXCLUDE_SUFFIXES:
                files.append(item)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Create arXiv source bundle")
    parser.add_argument(
        "--output",
        default="dist/arxiv_source.tar.gz",
        help="Output tar.gz path from repository root",
    )
    args = parser.parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = iter_included_files()
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(ROOT)))

    print(f"Created arXiv source bundle: {out_path}")
    print(f"Included files: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())