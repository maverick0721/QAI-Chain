from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "paper" / "figures"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _research_map(research: dict) -> dict[str, dict]:
    result_map: dict[str, dict] = {}
    for row in research.get("results", []):
        method = row["method"]
        agg = row.get("aggregate", {})
        per_seed = row.get("per_seed", [])
        result_map[method] = {
            "mean_reward_last5": float(agg.get("mean_reward_last5_mean", 0.0)),
            "std_reward_last5": float(agg.get("mean_reward_last5_std", 0.0)),
            "mean_abs_error_last5": float(agg.get("abs_target_error_mean", 0.0)),
            "per_seed_reward_last5_mean": [float(p.get("mean_reward_last5", 0.0)) for p in per_seed],
        }
    return result_map


def _detailed_last5_matrix(detailed: dict, method: str) -> np.ndarray:
    runs = detailed["methods"][method]
    arr = []
    for seed in sorted(runs.keys(), key=lambda s: int(s)):
        rewards = np.array(runs[seed]["episode_rewards"], dtype=np.float64)
        arr.append(float(rewards[-5:].mean()))
    return np.array(arr, dtype=np.float64)


def main() -> int:
    research = _load(RESULTS_DIR / "research_results.json")
    stats = _load(RESULTS_DIR / "statistical_analysis.json")
    detailed = _load(RESULTS_DIR / "detailed_results.json")

    rmap = _research_map(research)
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]

    for method in methods:
        if method not in rmap:
            raise RuntimeError(f"Missing method in research results: {method}")
        if method not in stats.get("summary", {}):
            raise RuntimeError(f"Missing method in statistical summary: {method}")
        if method not in detailed.get("methods", {}):
            raise RuntimeError(f"Missing method in detailed results: {method}")

    # Cross-artifact sanity checks.
    for method in methods:
        pub_mean = rmap[method]["mean_reward_last5"]
        pub_ci_lo = float(stats["summary"][method]["ci95_low"])
        pub_ci_hi = float(stats["summary"][method]["ci95_high"])
        if not (pub_ci_lo <= pub_mean <= pub_ci_hi):
            raise RuntimeError(f"Publication mean not within CI for {method}")

        detailed_last5 = _detailed_last5_matrix(detailed, method)
        if detailed_last5.shape[0] == 0:
            raise RuntimeError(f"No detailed last5 values for {method}")

    expected_figs = [
        "main_ablation_bar.png",
        "difficulty_error_bar.png",
        "component_latency.png",
        "bootstrap_ci.png",
        "learning_curves.png",
        "difficulty_curves.png",
        "seed_stability_boxplot.png",
        "reward_ecdf.png",
        "reward_error_tradeoff.png",
        "reward_heatmap.png",
        "reward_drift.png",
        "rank_stability_bootstrap.png",
        "adversarial_main_bar.png",
        "robust_ablation_bar.png",
        "robust_policy_flow.png",
        "adversarial_stress_sweep.png",
        "complexity_regime_sweep.png",
        "architecture_flow.png",
        "deployment_topology.png",
    ]

    missing = [name for name in expected_figs if not (FIG_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Missing generated figures: {missing}")

    print("Figure validation passed: data mappings are coherent and all expected files exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
