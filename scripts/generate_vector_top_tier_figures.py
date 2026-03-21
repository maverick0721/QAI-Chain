from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "paper" / "figures"


def load_json(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def save_pdf(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Wrote vector figure: {path}")


def fig_money_plot_ablation() -> None:
    data = load_json("full_ablation_matrix.json")["results"]
    methods = ["full_system", "classical_ppo", "cpo" if "cpo" in data else "no_quantum_uncertainty", "p3o" if "p3o" in data else "no_shield"]
    labels_map = {
        "full_system": "Robust (Ours)",
        "classical_ppo": "PPO",
        "no_quantum_uncertainty": "No Gate",
        "no_shield": "No Shield",
        "cpo": "CPO",
        "p3o": "P3O",
    }
    labels = [labels_map[m] for m in methods]

    reward = [float(data[m]["aggregate"]["reward_mean_mean"]) for m in methods]
    viol = [float(data[m]["aggregate"].get("safety_violations_mean_mean", data[m]["aggregate"].get("violations_mean_mean", 0.0))) for m in methods]
    mae = [float(data[m]["aggregate"].get("target_error_mae_mean_mean", data[m]["aggregate"].get("target_error_mean_mean", 0.0))) for m in methods]
    fallback = [float(data[m]["aggregate"].get("fallback_rate_mean_mean", 0.0)) for m in methods]

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.8))
    axs = axs.ravel()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    axs[0].bar(labels, reward, color=palette)
    axs[0].set_title("Reward (higher is better)")
    axs[0].tick_params(axis="x", rotation=15)
    axs[0].grid(axis="y", alpha=0.25)

    axs[1].bar(labels, viol, color=palette)
    axs[1].set_title("Safety Violations (lower is better)")
    axs[1].tick_params(axis="x", rotation=15)
    axs[1].grid(axis="y", alpha=0.25)

    axs[2].bar(labels, mae, color=palette)
    axs[2].set_title("Target MAE (lower is better)")
    axs[2].tick_params(axis="x", rotation=15)
    axs[2].grid(axis="y", alpha=0.25)

    axs[3].bar(labels, fallback, color=palette)
    axs[3].set_title("Fallback Rate")
    axs[3].tick_params(axis="x", rotation=15)
    axs[3].grid(axis="y", alpha=0.25)

    fig.suptitle("Ablation Money Plot (Vector)", y=1.02)
    save_pdf("money_plot_ablation.pdf")


def fig_stress_sweep() -> None:
    d = load_json("stress_sweep_scaled.json")["results"]
    xs = sorted(int(k.split("_")[1]) / 100.0 for k in d.keys())
    key_for = {int(k.split("_")[1]) / 100.0: k for k in d.keys()}

    methods = [
        ("ppo", "PPO", "#ff7f0e"),
        ("robust", "Robust (Ours)", "#1f77b4"),
        ("heuristic", "Heuristic", "#2ca02c"),
    ]

    plt.figure(figsize=(8.8, 4.8))
    for m, label, color in methods:
        ys = [float(d[key_for[x]][m]) for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=label)

    plt.xlabel("Adversarial Intensity")
    plt.ylabel("Mean Reward")
    plt.title("Stress Sweep: Reward vs Disturbance")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    save_pdf("stress_sweep_top_tier.pdf")


def fig_parameter_efficiency() -> None:
    matched = load_json("parameter_efficiency_matched.json")["results"]

    names = ["mlp_28", "vqc_6q_4l", "mlp_4673"]
    labels = ["MLP-28", "VQC-28", "MLP-4673"]
    xs = [float(matched[n]["params"]) for n in names]
    ys = [float(matched[n]["reward_mean"]) for n in names]
    cs = [float(matched[n]["violations_mean"]) for n in names]

    plt.figure(figsize=(8.8, 5.0))
    sc = plt.scatter(xs, ys, c=cs, cmap="viridis", s=[120, 140, 180], edgecolors="black", linewidths=0.8)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xscale("log")
    plt.xlabel("Parameter Count (log scale)")
    plt.ylabel("Mean Reward")
    plt.title("Parameter Efficiency Frontier")
    cbar = plt.colorbar(sc)
    cbar.set_label("Safety Violations")
    plt.grid(alpha=0.25)
    save_pdf("parameter_efficiency_pareto.pdf")


def fig_safety_reward_pareto() -> None:
    d = load_json("constrained_baseline_suite.json")["results"]["scaled"]
    rows = {
        "Robust (Ours)": d["robust_qgate_shield"]["aggregate"],
        "CPO": d["cpo"]["aggregate"],
        "P3O": d["p3o"]["aggregate"],
    }

    plt.figure(figsize=(7.8, 5.0))
    colors = {"Robust (Ours)": "#1f77b4", "CPO": "#d62728", "P3O": "#2ca02c"}
    for label, agg in rows.items():
        x = float(agg["violations_mean_mean"])
        y = float(agg["reward_mean_mean"])
        plt.scatter([x], [y], s=130, color=colors[label], edgecolors="black", linewidths=0.8)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)

    plt.xlabel("Safety Violations (lower is better)")
    plt.ylabel("Reward (higher is better)")
    plt.title("Pareto View: Safety Cost vs Reward")
    plt.grid(alpha=0.25)
    save_pdf("safety_reward_pareto.pdf")


def draw_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.0, edgecolor="#333333", facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10)


def arrow(ax, a: tuple[float, float], b: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=14, linewidth=1.3, color="#2b2b2b"))


def fig_robust_policy_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.03, 0.35, 0.18, 0.28, "State\nfeatures", "#e8f1ff")
    draw_box(ax, 0.27, 0.35, 0.18, 0.28, "Policy\naction", "#fff2e6")
    draw_box(ax, 0.50, 0.35, 0.20, 0.28, "Uncertainty\ngate", "#e9f7ef")
    draw_box(ax, 0.75, 0.35, 0.20, 0.28, "Shielded\nexecution", "#fef5e7")

    arrow(ax, (0.21, 0.49), (0.27, 0.49))
    arrow(ax, (0.45, 0.49), (0.50, 0.49))
    arrow(ax, (0.70, 0.49), (0.75, 0.49))

    ax.text(0.60, 0.19, "Fallback to deterministic heuristic when uncertainty > threshold", ha="center", fontsize=9)
    plt.title("Uncertainty-Gated Safe RL Control Flow")
    save_pdf("robust_policy_flow.pdf")


def fig_architecture_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.08, 0.68, 0.24, 0.18, "Policy\nproposal", "#e8f1ff")
    draw_box(ax, 0.38, 0.68, 0.24, 0.18, "Uncertainty\nestimator\n(compact VQC)", "#e9f7ef")
    draw_box(ax, 0.68, 0.68, 0.24, 0.18, "Shield\nexecution", "#fff2e6")
    draw_box(ax, 0.68, 0.34, 0.24, 0.18, "Verifiable\nlogging", "#fdf2f8")
    draw_box(ax, 0.38, 0.34, 0.24, 0.18, "Environment\nfeedback", "#f4f6f7")
    draw_box(ax, 0.08, 0.34, 0.24, 0.18, "Policy\nupdate", "#fef5e7")

    arrow(ax, (0.32, 0.77), (0.38, 0.77))
    arrow(ax, (0.62, 0.77), (0.68, 0.77))
    arrow(ax, (0.80, 0.68), (0.80, 0.52))
    arrow(ax, (0.68, 0.43), (0.62, 0.43))
    arrow(ax, (0.38, 0.43), (0.32, 0.43))
    arrow(ax, (0.20, 0.52), (0.20, 0.68))

    plt.title("Governance Control Loop with Verifiable Logging")
    save_pdf("architecture_loop_top_tier.pdf")


def main() -> int:
    fig_money_plot_ablation()
    fig_stress_sweep()
    fig_parameter_efficiency()
    fig_safety_reward_pareto()
    fig_robust_policy_flow()
    fig_architecture_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
