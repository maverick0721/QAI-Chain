from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    env = os.environ.copy()
    root_str = str(ROOT)
    current_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root_str}:{current_path}" if current_path else root_str
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def compile_pdf(paper_main: str) -> bool:
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None:
        print("pdflatex not found; skipping PDF compilation")
        return False
    if bibtex is None:
        print("bibtex not found; skipping PDF compilation")
        return False

    paper_dir = ROOT / "paper"
    stem = Path(paper_main).stem

    run([pdflatex, "-interaction=nonstopmode", paper_main], cwd=paper_dir)

    aux_path = paper_dir / f"{stem}.aux"
    aux_text = aux_path.read_text(encoding="utf-8") if aux_path.exists() else ""
    if "\\citation" in aux_text:
        run([bibtex, stem], cwd=paper_dir)
    else:
        print("No citation entries found; skipping bibtex")

    run([pdflatex, "-interaction=nonstopmode", paper_main], cwd=paper_dir)
    run([pdflatex, "-interaction=nonstopmode", paper_main], cwd=paper_dir)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build camera-ready artifacts")
    parser.add_argument(
        "--paper-main",
        default="main.tex",
        help="Paper entrypoint file inside paper/",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a reduced-scale pipeline for quick iteration.",
    )
    args = parser.parse_args()

    if args.fast:
        print("Running fast camera-ready pipeline...")
        run(
            [
                ".venv/bin/python",
                "experiments/run_constrained_baseline_suite.py",
                "--episodes",
                "12",
                "--steps",
                "80",
                "--seeds",
                "3,7,11,17",
                "--uncertainty-k",
                "2",
                "--uncertainty-every",
                "8",
            ],
            cwd=ROOT,
        )
        run([".venv/bin/python", "experiments/run_dual_environment_transfer.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_quantum_uncertainty_comparison.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_parameter_efficiency.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_stress_sweep_scaled.py", "--episodes", "8", "--steps", "100"], cwd=ROOT)
        run(
            [
                ".venv/bin/python",
                "experiments/run_full_ablation_matrix.py",
                "--episodes",
                "4",
                "--steps",
                "60",
                "--seeds",
                "3,7,11,17",
            ],
            cwd=ROOT,
        )
        run([".venv/bin/python", "scripts/run_reproducibility_harness.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_api_schema_docs.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_top_tier_artifacts.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/version_experiment_artifacts.py"], cwd=ROOT)
    else:
        print("Running full camera-ready pipeline...")
        run([".venv/bin/python", "experiments/run_publication_suite.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_adversarial_suite.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_complexity_regime_suite.py"], cwd=ROOT)
        run(
            [
                ".venv/bin/python",
                "experiments/run_constrained_baseline_suite.py",
                "--episodes",
                "20",
                "--steps",
                "120",
                "--uncertainty-k",
                "3",
                "--uncertainty-every",
                "6",
            ],
            cwd=ROOT,
        )
        run([".venv/bin/python", "experiments/run_dual_environment_transfer.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_quantum_uncertainty_comparison.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_parameter_efficiency.py"], cwd=ROOT)
        run([".venv/bin/python", "experiments/run_stress_sweep_scaled.py"], cwd=ROOT)
        run(
            [
                ".venv/bin/python",
                "experiments/run_full_ablation_matrix.py",
                "--episodes",
                "12",
                "--steps",
                "100",
                "--seeds",
                "3,7,11,17,21,42,84,126",
            ],
            cwd=ROOT,
        )
        run([".venv/bin/python", "scripts/analyze_research_results.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_benchmark_report.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_paper_tables.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/run_reproducibility_harness.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_api_schema_docs.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_adversarial_artifacts.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_adversarial_stress_sweep.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_complexity_artifacts.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/generate_top_tier_artifacts.py"], cwd=ROOT)
        run([".venv/bin/python", "scripts/version_experiment_artifacts.py"], cwd=ROOT)

    compiled = compile_pdf(args.paper_main)
    if compiled:
        print("camera-ready build complete: PDF compiled")
    else:
        print("camera-ready artifacts complete: PDF not compiled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())