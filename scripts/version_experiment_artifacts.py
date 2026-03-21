from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _git_revision() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = ROOT / "experiments" / "artifacts" / stamp
    version_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for src in [ROOT / "experiments" / "results", ROOT / "experiments" / "benchmarks"]:
        if not src.exists():
            continue
        dst = version_dir / src.name
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied.append(str(dst.relative_to(ROOT)))

    docs_subset = [
        "RESEARCH_RESULTS.md",
        "ADVERSARIAL_RESULTS.md",
        "COMPLEXITY_REGIME_RESULTS.md",
        "CONSTRAINED_BASELINE_SUITE.md",
        "FULL_ABLATION_MATRIX.md",
        "BENCHMARK_REPORT.md",
        "STATISTICAL_ANALYSIS.md",
    ]
    docs_dst = version_dir / "docs"
    docs_dst.mkdir(parents=True, exist_ok=True)
    for name in docs_subset:
        src = ROOT / "docs" / name
        if src.exists():
            shutil.copy2(src, docs_dst / name)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "artifact_root": str(version_dir.relative_to(ROOT)),
        "copied_paths": copied,
        "docs_subset": docs_subset,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "experiments" / "artifacts" / "latest.txt").write_text(stamp + "\n", encoding="utf-8")

    print(f"Versioned artifacts at: {version_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
