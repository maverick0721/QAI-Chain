# mypy: ignore-errors

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "paper" / "figures"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_traces(method_runs: dict[str, dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    traces = np.array([run[key] for run in method_runs.values()], dtype=np.float64)
    return traces.mean(axis=0), traces.std(axis=0)


def _research_map(research: dict) -> dict[str, dict]:
    if "methods" in research:
        return research["methods"]

    result_map: dict[str, dict] = {}
    for row in research.get("results", []):
        method = row["method"]
        agg = row.get("aggregate", {})
        per_seed = row.get("per_seed", [])
        result_map[method] = {
            "mean_reward_last5": agg.get("mean_reward_last5_mean", 0.0),
            "std_reward_last5": agg.get("mean_reward_last5_std", 0.0),
            "mean_abs_error_last5": agg.get("abs_target_error_mean", 0.0),
            "per_seed_reward_last5_mean": [
                p.get("mean_reward_last5", 0.0) for p in per_seed
            ],
        }
    return result_map


def _save(fig_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / fig_name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Wrote figure: {path}")


def _detailed_last5_matrix(detailed: dict, method: str) -> np.ndarray:
    runs = detailed["methods"][method]
    arr = []
    for seed in sorted(runs.keys(), key=lambda s: int(s)):
        rewards = np.array(runs[seed]["episode_rewards"], dtype=np.float64)
        arr.append(float(rewards[-5:].mean()))
    return np.array(arr, dtype=np.float64)


def figure_main_and_ablation(research: dict) -> None:
    rmap = _research_map(research)
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]
    labels = ["Random", "Heuristic", "PPO", "PPO w/o AdvNorm"]
    rewards = [float(rmap[m]["mean_reward_last5"]) for m in methods]
    stds = [float(rmap[m]["std_reward_last5"]) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(labels, rewards, yerr=stds, capsize=6, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylabel("Mean Reward (Last 5 Episodes)")
    ax.set_title("Main and Ablation Performance (Publication Protocol)")
    ax.grid(axis="y", alpha=0.25)
    _save("main_ablation_bar.png")


def figure_difficulty_error(research: dict) -> None:
    rmap = _research_map(research)
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]
    labels = ["Random", "Heuristic", "PPO", "PPO w/o AdvNorm"]
    errs = [float(rmap[m]["mean_abs_error_last5"]) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(
        labels,
        errs,
        color=["#4C78A8", "#F58518", "#54A24B", "#E45756"],
        edgecolor="#1f1f1f",
        linewidth=1.1,
    )
    ax.set_ylabel("Mean |difficulty - target|")
    ax.set_title("Difficulty Tracking Error (Publication Protocol)")
    ymax = max(errs) if errs else 1.0
    ax.set_ylim(0.0, ymax * 1.18 + 0.05)
    for idx, val in enumerate(errs):
        y = val + 0.03 if val > 0 else 0.03
        ax.text(idx, y, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        if val == 0.0:
            ax.scatter([idx], [0.0], color="#1f1f1f", s=26, zorder=4)
    ax.grid(axis="y", alpha=0.25)
    _save("difficulty_error_bar.png")


def figure_latency(bench: dict) -> None:
    if "components" in bench:
        components = [r["component"] for r in bench["components"]]
        latency = [float(r["mean_latency_ms"]) for r in bench["components"]]
    else:
        results = bench.get("results", {})
        components = list(results.keys())
        latency = [float(results[k]["avg_ms"]) for k in components]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(components, latency, color=["#72B7B2", "#B279A2", "#FF9DA6"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Component Latency Profile")
    ax.grid(axis="y", alpha=0.25)
    _save("component_latency.png")


def figure_ci(stats: dict) -> None:
    summary = stats.get("summary", {})
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]
    labels = ["Random", "Heuristic", "PPO", "PPO w/o AdvNorm"]
    lows = [float(summary[m]["ci95_low"]) for m in methods]
    highs = [float(summary[m]["ci95_high"]) for m in methods]
    mids = [(low + high) / 2.0 for low, high in zip(lows, highs)]
    errs = [(high - low) / 2.0 for low, high in zip(lows, highs)]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(labels))
    ax.errorbar(x, mids, yerr=errs, fmt="o", capsize=6, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Reward (95% CI)")
    ax.set_title("Method Reward Confidence Intervals (Publication Protocol)")
    ax.grid(axis="y", alpha=0.25)
    _save("bootstrap_ci.png")


def figure_episode_curves(detailed: dict) -> None:
    methods = detailed["methods"]
    chosen = [
        ("random", "Random", "#4C78A8"),
        ("heuristic_target", "Heuristic", "#F58518"),
        ("ppo_full", "PPO", "#54A24B"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for key, label, color in chosen:
        mean_t, std_t = _aggregate_traces(methods[key], "episode_rewards")
        x = np.arange(1, len(mean_t) + 1)
        ax.plot(x, mean_t, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean_t - std_t, mean_t + std_t, color=color, alpha=0.16)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves Across Episodes (Detailed Protocol)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save("learning_curves.png")


def figure_difficulty_curves(detailed: dict) -> None:
    methods = detailed["methods"]
    chosen = [
        ("random", "Random", "#4C78A8"),
        ("heuristic_target", "Heuristic", "#F58518"),
        ("ppo_full", "PPO", "#54A24B"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for key, label, color in chosen:
        mean_t, std_t = _aggregate_traces(methods[key], "episode_difficulties")
        x = np.arange(1, len(mean_t) + 1)
        ax.plot(x, mean_t, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean_t - std_t, mean_t + std_t, color=color, alpha=0.16)

    target = int(detailed["config"]["target_difficulty"])
    ax.axhline(target, linestyle="--", color="black", linewidth=1.1, label="Target")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Difficulty")
    ax.set_title("Difficulty Control Trajectories (Detailed Protocol)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save("difficulty_curves.png")


def figure_seed_distribution(detailed: dict, research: dict) -> None:
    _ = research
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]
    labels = ["Random", "Heuristic", "PPO", "PPO w/o AdvNorm"]

    per_seed_values = []
    for m in methods:
        vals = [float(v) for v in _detailed_last5_matrix(detailed, m)]
        per_seed_values.append(vals)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.boxplot(per_seed_values, tick_labels=labels, patch_artist=True)
    ax.set_ylabel("Per-seed Last-5 Mean Reward")
    ax.set_title("Cross-Seed Stability (Detailed Protocol)")
    ax.grid(axis="y", alpha=0.25)
    _save("seed_stability_boxplot.png")


def figure_reward_ecdf(detailed: dict) -> None:
    methods = [
        ("random", "Random", "#4C78A8"),
        ("heuristic_target", "Heuristic", "#F58518"),
        ("ppo_full", "PPO", "#54A24B"),
        ("ppo_no_adv_norm", "PPO w/o AdvNorm", "#E45756"),
    ]

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for key, label, color in methods:
        vals = np.sort(_detailed_last5_matrix(detailed, key))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, linewidth=2, label=label)

    ax.set_xlabel("Per-seed Last-5 Mean Reward")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Reward Distribution Across Seeds (Detailed Protocol)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save("reward_ecdf.png")


def figure_tradeoff_scatter(detailed: dict) -> None:
    methods = [
        ("random", "Random", "#4C78A8"),
        ("heuristic_target", "Heuristic", "#F58518"),
        ("ppo_full", "PPO", "#54A24B"),
        ("ppo_no_adv_norm", "PPO w/o AdvNorm", "#E45756"),
    ]
    target = int(detailed["config"]["target_difficulty"])

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for key, label, color in methods:
        runs = detailed["methods"][key]
        xs: list[float] = []
        ys: list[float] = []
        for seed in sorted(runs.keys(), key=lambda s: int(s)):
            rewards = np.array(runs[seed]["episode_rewards"], dtype=np.float64)
            diffs = np.array(runs[seed]["episode_difficulties"], dtype=np.float64)
            xs.append(float(rewards[-5:].mean()))
            ys.append(float(np.abs(diffs[-5:] - target).mean()))
        ax.scatter(xs, ys, s=42, color=color, alpha=0.8, label=label)

    ax.set_xlabel("Per-seed Last-5 Mean Reward")
    ax.set_ylabel("Per-seed Last-5 Mean |d - target|")
    ax.set_title("Reward vs Control-Error Tradeoff (Detailed Protocol)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save("reward_error_tradeoff.png")


def figure_reward_heatmap(detailed: dict) -> None:
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]
    labels = ["Random", "Heuristic", "PPO", "PPO w/o AdvNorm"]
    rows = []
    for m in methods:
        mean_t, _ = _aggregate_traces(detailed["methods"][m], "episode_rewards")
        rows.append(mean_t)
    arr = np.array(rows, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Episode")
    ax.set_title("Mean Reward Heatmap by Method (Detailed Protocol)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Episode Reward")
    _save("reward_heatmap.png")


def figure_reward_drift(detailed: dict) -> None:
    methods = [
        ("random", "Random", "#4C78A8"),
        ("heuristic_target", "Heuristic", "#F58518"),
        ("ppo_full", "PPO", "#54A24B"),
        ("ppo_no_adv_norm", "PPO w/o AdvNorm", "#E45756"),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for key, label, color in methods:
        mean_t, _ = _aggregate_traces(detailed["methods"][key], "episode_rewards")
        drift = np.diff(mean_t)
        smooth = np.convolve(drift, np.ones(5) / 5.0, mode="valid")
        x = np.arange(1, len(smooth) + 1)
        ax.plot(x, smooth, linewidth=2, color=color, label=label)

    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Episode Index (smoothed derivative)")
    ax.set_ylabel("Rolling Reward Drift")
    ax.set_title("Temporal Reward Drift (Detailed Protocol)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save("reward_drift.png")


def figure_rank_stability(detailed: dict) -> None:
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]
    labels = ["Random", "Heuristic", "PPO", "PPO w/o AdvNorm"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    method_vals = {m: _detailed_last5_matrix(detailed, m) for m in methods}
    n = next(iter(method_vals.values())).shape[0]
    rng = np.random.default_rng(42)
    reps = 5000
    win_counts = {m: 0 for m in methods}

    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        sample_means = {m: float(method_vals[m][idx].mean()) for m in methods}
        best = max(sample_means, key=sample_means.get)
        win_counts[best] += 1

    probs = [win_counts[m] / reps for m in methods]

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.bar(labels, probs, color=colors, edgecolor="#1f1f1f", linewidth=1.1, width=0.62)
    ax.set_ylim(0.0, 1.2)
    ax.set_ylabel("Probability of Ranking #1")
    ax.set_title("Method Ranking Stability via Bootstrap (Detailed Protocol)", pad=22)
    ax.margins(x=0.08)
    fig.subplots_adjust(top=0.86, bottom=0.16)
    for idx, val in enumerate(probs):
        if val >= 0.12:
            ax.text(idx, val - 0.08, f"{val:.2f}", ha="center", va="top", fontsize=10, color="white", weight="bold")
        else:
            ax.text(idx, 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#2b2b2b")
            if val == 0.0:
                ax.scatter([idx], [0.0], color="#1f1f1f", s=24, zorder=4)
    ax.grid(axis="y", alpha=0.25)
    _save("rank_stability_bootstrap.png")


def figure_architecture_flow() -> None:
    fig, ax = plt.subplots(figsize=(14.8, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.018,rounding_size=0.012")

    nodes = {
        "state": (0.06, 0.61, 0.14, 0.21, "Blockchain\nState + Events"),
        "proposal": (0.30, 0.61, 0.14, 0.21, "Policy\nProposal"),
        "gate": (0.54, 0.61, 0.14, 0.21, "Uncertainty\nGate"),
        "shield": (0.78, 0.61, 0.14, 0.21, "Shield\nValidation"),
        "fallback": (0.54, 0.16, 0.14, 0.21, "Deterministic\nFallback"),
        "commit": (0.78, 0.16, 0.14, 0.21, "On-chain\nAudit Commit"),
    }

    colors = {
        "state": "#C7D2E3",
        "proposal": "#D2E1DA",
        "gate": "#E4DBCF",
        "shield": "#DDD3E5",
        "fallback": "#E8DFC3",
        "commit": "#CFE0E2",
    }

    for k, (x, y, w, h, txt) in nodes.items():
        rect = FancyBboxPatch((x, y), w, h, **box_style, linewidth=1.8, facecolor=colors[k], edgecolor="#2A3F57", alpha=1.0, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color="#1f1f1f", fontsize=15, zorder=4)

    def edge_point(node: str, side: str) -> tuple[float, float]:
        x, y, w, h, _ = nodes[node]
        if side == "left":
            return (x, y + h / 2)
        if side == "right":
            return (x + w, y + h / 2)
        if side == "top":
            return (x + w / 2, y + h)
        return (x + w / 2, y)

    def connect(src: str, src_side: str, dst: str, dst_side: str, rad: float = 0.0, shrink_a: float = 4, shrink_b: float = 4) -> None:
        start = edge_point(src, src_side)
        end = edge_point(dst, dst_side)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.3,
            color="#303030",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            zorder=5,
        )
        ax.add_patch(arrow)

    connect("state", "right", "proposal", "left")
    connect("proposal", "right", "gate", "left")
    connect("gate", "right", "shield", "left")
    connect("gate", "bottom", "fallback", "top", shrink_a=6, shrink_b=6)
    connect("shield", "bottom", "commit", "top", shrink_a=6, shrink_b=6)
    connect("fallback", "right", "commit", "left")
    connect("shield", "bottom", "fallback", "right", rad=-0.26, shrink_a=6, shrink_b=6)

    ax.text(
        0.61,
        0.41,
        "high uncertainty",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#444444",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.8),
        zorder=5,
    )
    ax.text(
        0.73,
        0.11,
        "validated action",
        ha="left",
        va="center",
        fontsize=11.5,
        color="#444444",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.8),
        zorder=5,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture_flow.png", dpi=220)
    fig.savefig(FIG_DIR / "architecture_flow.pdf")
    plt.close(fig)
    print(f"Wrote figure: {FIG_DIR / 'architecture_flow.png'}")
    print(f"Wrote figure: {FIG_DIR / 'architecture_flow.pdf'}")


def figure_deployment_topology() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.02,rounding_size=0.02")
    nodes = {
        "clients": (0.03, 0.67, 0.17, 0.12, "Clients\nWallets + Dashboards"),
        "rpc": (0.25, 0.67, 0.17, 0.12, "RPC Gateway\nFastAPI"),
        "node_a": (0.55, 0.78, 0.16, 0.10, "Node A\nMiner + Mempool"),
        "node_b": (0.76, 0.78, 0.16, 0.10, "Node B\nPeer Replica"),
        "node_c": (0.66, 0.62, 0.16, 0.10, "Node C\nValidator"),
        "ai": (0.25, 0.43, 0.17, 0.12, "AI Service\nPPO + Encoder"),
        "quantum": (0.55, 0.43, 0.19, 0.12, "Quantum Service\nQNN/QTransformer"),
        "pqc": (0.79, 0.43, 0.16, 0.12, "PQC Service\nSign/Verify"),
        "artifacts": (0.03, 0.21, 0.21, 0.12, "Artifacts Store\nJSON + Figures + Tables"),
        "ci": (0.29, 0.21, 0.17, 0.12, "CI Pipeline\nTests + Build"),
    }

    colors = {
        "clients": "#4C78A8",
        "rpc": "#72B7B2",
        "node_a": "#54A24B",
        "node_b": "#54A24B",
        "node_c": "#54A24B",
        "ai": "#F58518",
        "quantum": "#B279A2",
        "pqc": "#E45756",
        "artifacts": "#9D755D",
        "ci": "#79706E",
    }

    for key, (x, y, w, h, txt) in nodes.items():
        rect = FancyBboxPatch((x, y), w, h, **box_style, linewidth=1.8, facecolor=colors[key], edgecolor="#1F1F1F", alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color="white", fontsize=10.8, weight="bold", zorder=4)

    def edge_point(node: str, side: str) -> tuple[float, float]:
        x, y, w, h, _ = nodes[node]
        if side == "left":
            return (x, y + h / 2)
        if side == "right":
            return (x + w, y + h / 2)
        if side == "top":
            return (x + w / 2, y + h)
        return (x + w / 2, y)

    def connect(src: str, src_side: str, dst: str, dst_side: str, rad: float = 0.0) -> None:
        start = edge_point(src, src_side)
        end = edge_point(dst, dst_side)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.9,
            color="#2e2e2e",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=8,
            shrinkB=8,
            zorder=2,
        )
        ax.add_patch(arrow)

    # Keep only essential arrows and route via box edges to avoid crossing labels.
    connect("clients", "right", "rpc", "left")
    connect("rpc", "right", "node_a", "left", rad=0.12)
    connect("node_a", "right", "node_b", "left")
    connect("node_b", "bottom", "node_c", "right", rad=0.15)
    connect("node_c", "top", "node_a", "bottom", rad=0.2)

    connect("rpc", "bottom", "ai", "top")
    connect("ai", "right", "quantum", "left")
    connect("quantum", "right", "pqc", "left")

    connect("ai", "left", "artifacts", "top", rad=0.26)
    connect("artifacts", "right", "ci", "left")
    connect("ci", "top", "rpc", "bottom", rad=-0.45)

    ax.text(0.5, 0.98, "QAI-Chain Deployment Topology", ha="center", va="center", fontsize=18, weight="bold", color="#1f1f1f")
    ax.text(
        0.5,
        0.946,
        "Service-level view with nodes, AI/quantum services, PQC modules, and reproducibility pipeline",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#3a3a3a",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    _save("deployment_topology.png")


def main() -> int:
    research = _load_json(RESULTS_DIR / "research_results.json")
    stats = _load_json(RESULTS_DIR / "statistical_analysis.json")
    detailed = _load_json(RESULTS_DIR / "detailed_results.json")
    bench = _load_json(ROOT / "experiments" / "benchmarks" / "latest.json")

    figure_main_and_ablation(research)
    figure_difficulty_error(research)
    figure_latency(bench)
    figure_ci(stats)
    figure_episode_curves(detailed)
    figure_difficulty_curves(detailed)
    figure_seed_distribution(detailed, research)
    figure_reward_ecdf(detailed)
    figure_tradeoff_scatter(detailed)
    figure_reward_heatmap(detailed)
    figure_reward_drift(detailed)
    figure_rank_stability(detailed)
    figure_architecture_flow()
    figure_deployment_topology()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())