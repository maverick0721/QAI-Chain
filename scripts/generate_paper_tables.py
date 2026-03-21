from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_label(method: str) -> str:
    mapping = {
        "random": "Random",
        "heuristic_target": "Heuristic Target",
        "ppo_full": "PPO (Full)",
        "ppo_no_entropy_anneal": "PPO w/o Entropy Anneal",
        "ppo_no_adv_norm": "PPO w/o Advantage Norm",
    }
    return mapping.get(method, method)


def _render_main_table(results_payload: dict) -> str:
    rows = []
    for row in results_payload["results"]:
        method = row["method"]
        if method not in {"random", "heuristic_target", "ppo_full"}:
            continue
        agg = row["aggregate"]
        rows.append(
            "{} & {:.3f} $\\pm$ {:.3f} & {:.3f} \\\\".format(
                _method_label(method),
                agg["mean_reward_last5_mean"],
                agg["mean_reward_last5_std"],
                agg["abs_target_error_mean"],
            )
        )

    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Mean Reward (Last 5) & Final $|d-d_t|$ " + "\\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _render_ablation_table(results_payload: dict) -> str:
    rows = []
    for row in results_payload["results"]:
        method = row["method"]
        if not method.startswith("ppo"):
            continue
        agg = row["aggregate"]
        rows.append(
            "{} & {:.3f} $\\pm$ {:.3f} \\\\".format(
                _method_label(method),
                agg["mean_reward_last5_mean"],
                agg["mean_reward_last5_std"],
            )
        )

    lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Variant & Mean Reward (Last 5) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _render_benchmark_table(benchmark_payload: dict) -> str:
    rows = []
    for component, stats in benchmark_payload["results"].items():
        rows.append(
            "{} & {:.3f} & {:.3f} & {:.3f} & [{:.3f}, {:.3f}] \\\\".format(
                component,
                stats["avg_ms"],
                stats["p50_ms"],
                stats["p90_ms"],
                stats["ci95_low_ms"],
                stats["ci95_high_ms"],
            )
        )

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Component & Avg (ms) & P50 (ms) & P90 (ms) & 95\\% CI (ms) " + "\\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _render_stats_table(stats_payload: dict) -> str:
    rows = []
    for comp, vals in stats_payload["comparisons"].items():
        rows.append(
            "{} & {:.4f} & {:.3f} \\\\".format(
                comp.replace("_", " "),
                vals["permutation_pvalue"],
                vals["cohens_d"],
            )
        )

    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Comparison & Permutation p-value & Cohen's d \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    results_path = ROOT / "experiments" / "results" / "research_results.json"
    benchmark_path = ROOT / "experiments" / "benchmarks" / "latest.json"
    stats_path = ROOT / "experiments" / "results" / "statistical_analysis.json"

    if not results_path.exists() or not benchmark_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            "Required artifacts missing. Run experiments/run_research_suite.py "
            "plus scripts/generate_benchmark_report.py and "
            "scripts/analyze_research_results.py first."
        )

    results_payload = _load_json(results_path)
    benchmark_payload = _load_json(benchmark_path)
    stats_payload = _load_json(stats_path)

    tables_dir = ROOT / "paper" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    (tables_dir / "main_comparison.tex").write_text(
        _render_main_table(results_payload),
        encoding="utf-8",
    )
    (tables_dir / "ablation.tex").write_text(
        _render_ablation_table(results_payload),
        encoding="utf-8",
    )
    (tables_dir / "benchmark.tex").write_text(
        _render_benchmark_table(benchmark_payload),
        encoding="utf-8",
    )
    (tables_dir / "statistics.tex").write_text(
        _render_stats_table(stats_payload),
        encoding="utf-8",
    )

    print(f"Wrote LaTeX tables to: {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())