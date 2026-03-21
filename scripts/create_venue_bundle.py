from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

VENUE_CONFIG = {
    "neurips": {
        "paper_main": "main_neurips_style.tex",
        "checklist": "docs/NEURIPS_SUBMISSION_CHECKLIST.md",
    },
    "iclr": {
        "paper_main": "main_iclr_style.tex",
        "checklist": "docs/ICLR_SUBMISSION_CHECKLIST.md",
    },
    "ieee": {
        "paper_main": "main_ieee_style.tex",
        "checklist": "docs/IEEE_SUBMISSION_CHECKLIST.md",
    },
}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create venue-specific submission bundle")
    parser.add_argument("--venue", choices=["neurips", "iclr", "ieee", "all"], required=True)
    parser.add_argument(
        "--skip-camera-ready",
        action="store_true",
        help="Skip artifact regeneration and paper build pipeline",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    venues = list(VENUE_CONFIG.keys()) if args.venue == "all" else [args.venue]

    for venue in venues:
        cfg = VENUE_CONFIG[venue]
        paper_main = cfg["paper_main"]

        if not args.skip_camera_ready:
            run([
                sys.executable,
                "scripts/build_camera_ready.py",
                "--paper-main",
                paper_main,
            ])

        out_name = f"dist/qai-chain-{venue}-source-{timestamp}.tar.gz"
        run([sys.executable, "scripts/create_arxiv_bundle.py", "--output", out_name])

        checklist = cfg["checklist"]
        print("")
        print(f"Venue bundle ready: {ROOT / out_name}")
        print(f"Submission checklist: {ROOT / checklist}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())