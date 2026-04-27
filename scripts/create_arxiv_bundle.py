from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Paths included relative to repository root (upload tarball root).
INCLUDE_PATHS = [
    "00README.yaml",
    "paper",
    "docs/BENCHMARK_REPORT.md",
    "docs/RESEARCH_RESULTS.md",
    "docs/STATISTICAL_ANALYSIS.md",
    "experiments/results/research_results.json",
    "experiments/results/statistical_analysis.json",
    "experiments/benchmarks/latest.json",
]

# Auxiliary TeX outputs to omit (keep .bbl for reproducible BibTeX on arXiv).
EXCLUDE_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".toc",
}

# arXiv compiles from source; do not ship the local PDF next to main.tex.
EXCLUDE_RELATIVE_PATHS = {
    Path("paper/main.pdf"),
}

# https://info.arxiv.org/help/submit/index.html — allowed characters in file names
_ARXIV_NAME_CHARS = re.compile(r"^[a-zA-Z0-9_+\-.,=]+$")


def _path_segments_ok(rel: Path) -> bool:
    for part in rel.parts:
        if not _ARXIV_NAME_CHARS.match(part):
            return False
    return True


def iter_included_files() -> list[Path]:
    files: list[Path] = []
    bad_names: list[str] = []

    for rel in INCLUDE_PATHS:
        p = ROOT / rel
        if not p.exists():
            continue
        if p.is_file():
            rel_path = p.relative_to(ROOT)
            if not _path_segments_ok(rel_path):
                bad_names.append(str(rel_path))
            else:
                files.append(p)
            continue
        for item in p.rglob("*"):
            if not item.is_file():
                continue
            if item.suffix in EXCLUDE_SUFFIXES:
                continue
            rel_path = item.relative_to(ROOT)
            if rel_path in EXCLUDE_RELATIVE_PATHS:
                continue
            if not _path_segments_ok(rel_path):
                bad_names.append(str(rel_path))
                continue
            files.append(item)

    if bad_names:
        print(
            "ERROR: file names contain characters arXiv does not allow "
            "(only a-z A-Z 0-9 _ + - . , =):\n  "
            + "\n  ".join(sorted(set(bad_names))),
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Stable order for reproducible tarballs
    files.sort(key=lambda x: str(x.relative_to(ROOT)).lower())
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

    readme = ROOT / "00README.yaml"
    if not readme.is_file():
        print("WARNING: missing 00README.yaml at repo root (compiler hint).", file=sys.stderr)

    bbl = ROOT / "paper" / "main.bbl"
    if not bbl.is_file():
        print(
            "WARNING: paper/main.bbl not found. Run bibtex main from paper/ before bundling "
            "so arXiv uses a pre-built bibliography.",
            file=sys.stderr,
        )

    files = iter_included_files()
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(ROOT)))

    print(f"Created arXiv source bundle: {out_path}")
    print(f"Included files: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
