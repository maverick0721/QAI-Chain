from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run seed-controlled reproducibility harness")
    parser.add_argument("--seeds", default="3,7,11,17", help="Comma-separated seed list")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    _run(
        [
            ".venv/bin/python",
            "experiments/run_constrained_baseline_suite.py",
            "--seeds",
            args.seeds,
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.steps),
            "--uncertainty-k",
            "2",
            "--uncertainty-every",
            "8",
        ]
    )

    result_path = ROOT / "experiments" / "results" / "constrained_baseline_suite.json"
    report_path = ROOT / "docs" / "REPRODUCIBILITY_HARNESS.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [int(x) for x in args.seeds.split(",") if x.strip()],
        "episodes": args.episodes,
        "steps": args.steps,
        "artifact": str(result_path.relative_to(ROOT)),
        "sha256": _sha256(result_path),
    }

    out_json = ROOT / "experiments" / "results" / "reproducibility_harness.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Reproducibility Harness",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        f"- Seeds: {payload['seeds']}",
        f"- Episodes: {payload['episodes']}",
        f"- Steps: {payload['steps']}",
        f"- Artifact: {payload['artifact']}",
        f"- SHA256: {payload['sha256']}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote reproducibility artifact: {out_json}")
    print(f"Wrote reproducibility report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
