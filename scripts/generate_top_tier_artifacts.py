from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    out = FIG / name
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"Wrote figure: {out}")


def fig_architecture_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.axis("off")
    nodes = [
        (0.06, 0.42, 0.18, 0.18, "Blockchain\nState", "#67A9CF"),
        (0.30, 0.42, 0.18, 0.18, "Quantum\nUncertainty", "#2C7FB8"),
        (0.54, 0.42, 0.18, 0.18, "Robust PPO", "#F28E2B"),
        (0.78, 0.42, 0.18, 0.18, "Safety\nShield", "#59A14F"),
    ]
    for x, y, w, h, label, color in nodes:
        rect = plt.Rectangle((x, y), w, h, color=color, alpha=0.9, ec="#1f1f1f", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    ax.annotate("", xy=(0.30, 0.51), xytext=(0.24, 0.51), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.54, 0.51), xytext=(0.48, 0.51), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.78, 0.51), xytext=(0.72, 0.51), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.15, 0.37), xytext=(0.87, 0.37), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="arc3,rad=-0.35"))
    ax.text(0.50, 0.22, "On-Chain Commit + Next Epoch", ha="center", va="center", fontsize=11)
    ax.set_title("Quantum-Gated Safe Governance Loop", fontsize=14, fontweight="bold")
    _save("architecture_loop_top_tier.png")


def fig_money_plot(ablation: dict) -> None:
    r = ablation["results"]
    full = r["full_system"]["aggregate"]
    abls = ["no_quantum_uncertainty", "no_shield", "no_fallback"]

    labels = ["Full", "-QGate", "-Shield", "-Fallback"]
    reward = [full["reward_mean_mean"]] + [r[k]["aggregate"]["reward_mean_mean"] for k in abls]
    viol = [full["safety_violations_mean_mean"]] + [r[k]["aggregate"]["safety_violations_mean_mean"] for k in abls]
    mae = [full["target_error_mae_mean_mean"]] + [r[k]["aggregate"]["target_error_mae_mean_mean"] for k in abls]
    fb = [full["fallback_rate_mean_mean"]] + [r[k]["aggregate"]["fallback_rate_mean_mean"] for k in abls]

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2))
    colors = ["#2E86AB", "#F6C85F", "#6F4E7C", "#9FD356"]
    axs[0, 0].bar(labels, reward, color=colors)
    axs[0, 0].set_title("Reward")
    axs[0, 1].bar(labels, viol, color=colors)
    axs[0, 1].set_title("Safety Violations")
    axs[1, 0].bar(labels, mae, color=colors)
    axs[1, 0].set_title("Target MAE")
    axs[1, 1].bar(labels, fb, color=colors)
    axs[1, 1].set_title("Fallback Rate")
    for ax in axs.ravel():
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Main Ablation Result (Money Plot)", fontsize=14, fontweight="bold")
    _save("money_plot_ablation.png")


def fig_stress_sweep(stress: dict) -> None:
    keys = sorted(stress["results"].keys(), key=lambda x: int(x.split("_")[-1]))
    x = [int(k.split("_")[-1]) for k in keys]
    methods = ["heuristic", "ppo", "robust", "random"]
    colors = {"heuristic": "#E15759", "ppo": "#4E79A7", "robust": "#59A14F", "random": "#999999"}

    plt.figure(figsize=(9.5, 4.8))
    for m in methods:
        y = [stress["results"][k][m] for k in keys]
        plt.plot(x, y, marker="o", linewidth=2.0, label=m, color=colors[m])
    plt.xlabel("Adversarial Intensity (%)")
    plt.ylabel("Mean Reward")
    plt.title("Stress Sweep with Crossover Region")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    _save("stress_sweep_top_tier.png")


def fig_parameter_pareto(param_eff: dict) -> None:
    plt.figure(figsize=(8.8, 4.8))
    if "results" in param_eff:
        c = param_eff["results"]["classical_mlp"]
        q = param_eff["results"]["vqc_6q_4l"]
        plt.scatter([c["params"]], [c["reward_mean"]], color="#E15759", s=90, label="Classical MLP")
        plt.scatter([q["params"]], [q["reward_mean"]], color="#4E79A7", s=90, label="VQC")
    else:
        c = param_eff["classical"]
        q = param_eff["vqc"]
        plt.scatter([r["params"] for r in c], [r["reward"] for r in c], color="#E15759", s=70, label="Classical MLP")
        plt.scatter([r["params"] for r in q], [r["reward"] for r in q], color="#4E79A7", s=70, label="VQC")
    plt.xscale("log")
    plt.xlabel("Parameter Count (log scale)")
    plt.ylabel("Mean Reward")
    plt.title("Parameter Efficiency Pareto Frontier")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    _save("parameter_efficiency_pareto.png")


def table_baselines(baselines: dict) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Env & Method & Reward & Reward std & Violations & Target MAE & Fallback \\",
        r"\midrule",
    ]
    order = ["random", "heuristic", "cpo", "p3o", "ppo_lagrangian", "robust_qgate_shield"]
    names = {
        "random": "Random",
        "heuristic": "Heuristic",
        "cpo": "CPO-style",
        "p3o": "P3O-style",
        "ppo_lagrangian": "PPO-Lagrangian-style",
        "robust_qgate_shield": "Robust QGate+Shield",
    }
    env_labels = {"scaled": "Scaled", "defi": "DeFi"}
    for env_name in ["scaled", "defi"]:
        for idx, k in enumerate(order):
            a = baselines["results"][env_name][k]["aggregate"]
            env_cell = env_labels[env_name] if idx == 0 else ""
            lines.append(
                f"{env_cell} & {names[k]} & {a['reward_mean_mean']:.2f} & {a['reward_mean_std']:.2f} & "
                f"{a['violations_mean_mean']:.2f} & {a['target_error_mean_mean']:.3f} & {a['fallback_rate_mean_mean']:.3f} \\\\" 
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out = TAB / "constrained_baseline_comparison.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote table: {out}")


def main() -> int:
    ablation = _load(RESULTS / "full_ablation_matrix.json")
    stress = _load(RESULTS / "stress_sweep_scaled.json")
    param_eff = _load(RESULTS / "parameter_efficiency.json")
    baselines = _load(RESULTS / "constrained_baseline_suite.json")

    fig_architecture_loop()
    fig_money_plot(ablation)
    fig_stress_sweep(stress)
    fig_parameter_pareto(param_eff)
    table_baselines(baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
