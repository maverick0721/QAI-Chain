from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * (n_boot - 1))
    hi_idx = int((1 - alpha / 2) * (n_boot - 1))
    return float(means[lo_idx]), float(means[hi_idx])


def _permutation_pvalue(a: list[float], b: list[float], n_perm: int = 20000) -> float:
    observed = abs(statistics.mean(a) - statistics.mean(b))
    combined = a + b
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        random.shuffle(combined)
        aa = combined[:n_a]
        bb = combined[n_a:]
        diff = abs(statistics.mean(aa) - statistics.mean(bb))
        if diff >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def _cohens_d(a: list[float], b: list[float]) -> float:
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.pvariance(a) if len(a) > 1 else 0.0
    var_b = statistics.pvariance(b) if len(b) > 1 else 0.0
    pooled = ((var_a + var_b) / 2) ** 0.5
    if pooled == 0:
        return 0.0
    return (mean_a - mean_b) / pooled


def main() -> int:
    random.seed(12345)

    results_path = ROOT / "experiments" / "results" / "research_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            "Missing research results. "
            "Run experiments/run_publication_suite.py first."
        )

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload["results"]

    method_to_values: dict[str, list[float]] = {}
    for row in rows:
        method = row["method"]
        vals = [x["mean_reward_last5"] for x in row["per_seed"]]
        method_to_values[method] = vals

    summary = {}
    for method, vals in method_to_values.items():
        ci_lo, ci_hi = _bootstrap_ci(vals)
        summary[method] = {
            "mean": float(statistics.mean(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "n": len(vals),
        }

    ppo_vals = method_to_values["ppo_full"]
    random_vals = method_to_values["random"]
    heuristic_vals = method_to_values["heuristic_target"]

    comparisons = {
        "ppo_vs_random": {
            "permutation_pvalue": _permutation_pvalue(ppo_vals, random_vals),
            "cohens_d": _cohens_d(ppo_vals, random_vals),
        },
        "ppo_vs_heuristic": {
            "permutation_pvalue": _permutation_pvalue(ppo_vals, heuristic_vals),
            "cohens_d": _cohens_d(ppo_vals, heuristic_vals),
        },
    }

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "comparisons": comparisons,
    }

    out_json = ROOT / "experiments" / "results" / "statistical_analysis.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Statistical Analysis",
        "",
        f"Generated (UTC): {out['generated_at_utc']}",
        "",
        "## 95% CI (bootstrap) for Mean Reward (Last5)",
        "",
        "| Method | Mean | Std | 95% CI Low | 95% CI High | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, stats in summary.items():
        lines.append(
            "| "
            f"{method} | {stats['mean']:.3f} | {stats['std']:.3f} | "
            f"{stats['ci95_low']:.3f} | {stats['ci95_high']:.3f} | {stats['n']} |"
        )

    lines.extend(
        [
            "",
            "## Permutation Tests (Mean Reward Last5)",
            "",
            "| Comparison | p-value | Cohen's d |",
            "|---|---:|---:|",
        ]
    )
    for comp, comp_stats in comparisons.items():
        lines.append(
            "| "
            f"{comp} | {comp_stats['permutation_pvalue']:.4f} | "
            f"{comp_stats['cohens_d']:.3f} |"
        )

    out_md = ROOT / "docs" / "STATISTICAL_ANALYSIS.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote statistical analysis JSON: {out_json}")
    print(f"Wrote statistical analysis markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())