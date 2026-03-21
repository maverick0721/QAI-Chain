from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IN_JSON = ROOT / "experiments" / "results" / "complexity_regime_results.json"
OUT_TABLE = ROOT / "paper" / "tables" / "complexity_regime_summary.tex"
OUT_FIG = ROOT / "paper" / "figures" / "complexity_regime_sweep.png"


def main() -> int:
    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    results = payload["results"]

    levels = sorted(results.keys(), key=lambda x: float(x.split("_")[-1]))
    names = {
        "heuristic_resilience": "Heuristic Resilience",
        "ppo_full": "PPO",
        "robust_ppo": "Robust PPO (Safe)",
    }
    colors = {
        "heuristic_resilience": "#4C78A8",
        "ppo_full": "#E45756",
        "robust_ppo": "#54A24B",
    }

    # Table summarizing reward at each complexity level.
    table = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & C=0.6 & C=1.0 & C=1.4 & C=1.8 \\",
        r"\midrule",
    ]

    by_method: dict[str, list[tuple[float, float]]] = {
        "heuristic_resilience": [],
        "ppo_full": [],
        "robust_ppo": [],
    }

    for lvl in levels:
        rows = {row["method"]: row for row in results[lvl]}
        for method in by_method:
            agg = rows[method]["aggregate"]
            by_method[method].append(
                (
                    float(agg["mean_reward_last5_mean"]),
                    float(agg["mean_reward_last5_std"]),
                )
            )

    for method, vals in by_method.items():
        row = [names[method]]
        for m, s in vals:
            row.append(f"{m:.2f} $\\pm$ {s:.2f}")
        table.append(" & ".join(row) + r" \\")

    table += [r"\bottomrule", r"\end{tabular}"]
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text("\n".join(table) + "\n", encoding="utf-8")

    # Sweep figure
    xs = [float(x.split("_")[-1]) for x in levels]
    plt.figure(figsize=(9.0, 4.8))
    for method in ["heuristic_resilience", "ppo_full", "robust_ppo"]:
        means = [m for m, _ in by_method[method]]
        stds = [s for _, s in by_method[method]]
        plt.plot(xs, means, marker="o", linewidth=2.0, color=colors[method], label=names[method])
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        plt.fill_between(xs, lo, hi, color=colors[method], alpha=0.15)

    plt.title("Complexity Regime Sweep: Learned vs Fixed Controllers")
    plt.xlabel("Complexity Level")
    plt.ylabel("Mean Reward (Last 5 Episodes)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=220)
    plt.close()

    print(f"Wrote complexity table: {OUT_TABLE}")
    print(f"Wrote complexity figure: {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
