from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
IN_JSON = ROOT / "experiments" / "results" / "adversarial_results.json"
OUT_TABLE = ROOT / "paper" / "tables" / "adversarial_comparison.tex"
OUT_SIG_TABLE = ROOT / "paper" / "tables" / "adversarial_significance.tex"
OUT_ROBUST_ABLATION = ROOT / "paper" / "tables" / "robust_ablation.tex"
OUT_FIG = ROOT / "paper" / "figures" / "adversarial_main_bar.png"
OUT_ABLATION_FIG = ROOT / "paper" / "figures" / "robust_ablation_bar.png"
OUT_POLICY_FIG = ROOT / "paper" / "figures" / "robust_policy_flow.png"


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
    random.seed(20260319)

    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    rows = payload["results"]

    display_name = {
        "random": "Random",
        "heuristic_target": "Heuristic Target",
        "heuristic_resilience": "Heuristic Resilience",
        "ppo_full": "PPO",
        "robust_risk_only": "PPO + Risk",
        "robust_fallback_only": "PPO + Fallback",
        "robust_ppo": "R-PPO (Safe)",
    }
    palette = {
        "random": "#9D9D9D",
        "heuristic_target": "#F58518",
        "heuristic_resilience": "#4C78A8",
        "ppo_full": "#E45756",
        "robust_risk_only": "#B279A2",
        "robust_fallback_only": "#72B7B2",
        "robust_ppo": "#54A24B",
    }

    labels = []
    rewards = []
    reward_std = []
    delays = []
    forks = []
    target_errors = []
    fallback_rates = []
    colors = []
    per_seed_last5: dict[str, list[float]] = {}

    core_methods = {
        "random",
        "heuristic_target",
        "heuristic_resilience",
        "ppo_full",
        "robust_ppo",
    }

    for row in rows:
        method = row["method"]
        agg = row["aggregate"]
        if method in core_methods:
            labels.append(display_name.get(method, method.replace("_", " ").title()))
            rewards.append(float(agg["mean_reward_last5_mean"]))
            reward_std.append(float(agg["mean_reward_last5_std"]))
            delays.append(float(agg["mean_delay_last5_mean"]))
            forks.append(float(agg["mean_fork_pressure_last5_mean"]))
            target_errors.append(float(agg["mean_target_error_last5_mean"]))
            fallback_rates.append(agg.get("mean_fallback_rate_last5_mean"))
            colors.append(palette.get(method, "#4C78A8"))
        per_seed_last5[method] = [float(x["mean_reward_last5"]) for x in row.get("per_seed", [])]

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table = [
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Reward (Last5) & Delay & Fork & Error & Fallback \\",
        r"\midrule",
    ]

    for i, label in enumerate(labels):
        fallback = fallback_rates[i]
        fallback_text = f"{float(fallback):.3f}" if fallback is not None else "-"
        table.append(
            f"{label} & {rewards[i]:.3f} $\\pm$ {reward_std[i]:.3f} & {delays[i]:.3f} & {forks[i]:.3f} & {target_errors[i]:.3f} & {fallback_text} \\\\"  # noqa: E501
        )

    table += [r"\bottomrule", r"\end{tabular}"]
    OUT_TABLE.write_text("\n".join(table) + "\n", encoding="utf-8")

    rows_by_method = {row["method"]: row for row in rows}
    ablation_methods = [
        "ppo_full",
        "robust_risk_only",
        "robust_fallback_only",
        "robust_ppo",
    ]
    ablation_table = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Variant & Reward (Last5) & Fallback Rate \\",
        r"\midrule",
    ]
    abl_labels = []
    abl_rewards = []
    abl_stds = []
    for method in ablation_methods:
        row = rows_by_method.get(method)
        if row is None:
            continue
        agg = row["aggregate"]
        label = display_name.get(method, method)
        abl_labels.append(label)
        abl_rewards.append(float(agg["mean_reward_last5_mean"]))
        abl_stds.append(float(agg["mean_reward_last5_std"]))
        ablation_table.append(
            f"{label} & {agg['mean_reward_last5_mean']:.3f} $\\pm$ {agg['mean_reward_last5_std']:.3f} & {agg['mean_fallback_rate_last5_mean']:.3f} \\\\"  # noqa: E501
        )
    ablation_table += [r"\bottomrule", r"\end{tabular}"]
    OUT_ROBUST_ABLATION.write_text("\n".join(ablation_table) + "\n", encoding="utf-8")

    robust = per_seed_last5.get("robust_ppo", [])
    heur = per_seed_last5.get("heuristic_resilience", [])
    ppo = per_seed_last5.get("ppo_full", [])
    sig_rows = []
    if robust and heur:
        sig_rows.append(
            (
                "Robust PPO (Safe) vs Heuristic Resilience",
                _permutation_pvalue(robust, heur),
                _cohens_d(robust, heur),
            )
        )
    if robust and ppo:
        sig_rows.append(
            (
                "Robust PPO (Safe) vs PPO",
                _permutation_pvalue(robust, ppo),
                _cohens_d(robust, ppo),
            )
        )

    sig_table = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Comparison & Permutation p-value & Cohen's d \\",
        r"\midrule",
    ]
    for name, pval, eff in sig_rows:
        sig_table.append(f"{name} & {pval:.4f} & {eff:.3f} \\\\")
    sig_table += [r"\bottomrule", r"\end{tabular}"]
    OUT_SIG_TABLE.write_text("\n".join(sig_table) + "\n", encoding="utf-8")

    plt.figure(figsize=(9.2, 4.6))
    plt.bar(labels, rewards, yerr=reward_std, capsize=6, color=colors, edgecolor="#1f1f1f", linewidth=1.0)
    plt.title("Adversarial Benchmark: Reward Under Stochastic Disturbances")
    plt.ylabel("Mean Reward (Last 5 Episodes)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=220)
    plt.close()

    plt.figure(figsize=(8.0, 4.2))
    ablation_colors = [palette.get("ppo_full"), palette.get("robust_risk_only"), palette.get("robust_fallback_only"), palette.get("robust_ppo")]
    plt.bar(abl_labels, abl_rewards, yerr=abl_stds, capsize=5, color=ablation_colors, edgecolor="#1f1f1f", linewidth=1.0)
    plt.title("Robust PPO Ablation in Adversarial Benchmark")
    plt.ylabel("Mean Reward (Last 5 Episodes)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    OUT_ABLATION_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_ABLATION_FIG, dpi=220)
    plt.close()

    # Robust policy intuition diagram.
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = {
        "state": (0.03, 0.52, 0.2, 0.28, "State + Metrics"),
        "ppo": (0.30, 0.60, 0.2, 0.24, "PPO Policy\n(mean, std)"),
        "gate": (0.56, 0.58, 0.2, 0.26, "Uncertainty Gate\n$\\sigma_t > \\tau$?"),
        "fallback": (0.30, 0.18, 0.22, 0.22, "Resilience\nHeuristic"),
        "risk": (0.56, 0.18, 0.2, 0.22, "Risk Penalty\n$\\tilde r_t$"),
        "action": (0.81, 0.40, 0.16, 0.28, "Safe Action\n+ PPO Update"),
    }
    for _, (x, y, w, h, text) in boxes.items():
        rect = plt.Rectangle((x, y), w, h, facecolor="#F8F9FA", edgecolor="#1f1f1f", linewidth=1.45)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.8)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.75, color="#111111", mutation_scale=13),
        )

    arrow(0.23, 0.66, 0.30, 0.72)
    arrow(0.50, 0.72, 0.56, 0.71)
    arrow(0.23, 0.56, 0.30, 0.29)
    arrow(0.52, 0.29, 0.56, 0.29)
    arrow(0.76, 0.71, 0.81, 0.59)
    arrow(0.76, 0.29, 0.81, 0.50)
    ax.text(
        0.66,
        0.88,
        "High uncertainty -> fallback",
        fontsize=8.8,
        color="#202020",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#F2F3F5", edgecolor="#989898", linewidth=1.0),
    )
    ax.text(
        0.66,
        0.06,
        "Low uncertainty -> PPO + risk-aware update",
        fontsize=8.8,
        color="#202020",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#F2F3F5", edgecolor="#989898", linewidth=1.0),
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.10)
    OUT_POLICY_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_POLICY_FIG, dpi=220)
    plt.close()

    print(f"Wrote table: {OUT_TABLE}")
    print(f"Wrote robust ablation table: {OUT_ROBUST_ABLATION}")
    print(f"Wrote significance table: {OUT_SIG_TABLE}")
    print(f"Wrote figure: {OUT_FIG}")
    print(f"Wrote ablation figure: {OUT_ABLATION_FIG}")
    print(f"Wrote policy diagram: {OUT_POLICY_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
