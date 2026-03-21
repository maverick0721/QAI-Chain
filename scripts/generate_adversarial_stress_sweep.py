from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_adversarial_suite import Config, run_baseline, run_ppo

OUT_JSON = ROOT / "experiments" / "results" / "adversarial_stress_sweep.json"
OUT_FIG = ROOT / "paper" / "figures" / "adversarial_stress_sweep.png"


def _scaled_cfg(base: Config, stress_scale: float) -> Config:
    return Config(
        episodes=34,
        steps_per_episode=24,
        seeds=(3, 7, 11, 17),
        start_difficulty=base.start_difficulty,
        target_difficulty=base.target_difficulty,
        min_difficulty=base.min_difficulty,
        max_difficulty=base.max_difficulty,
        attack_prob=min(0.95, base.attack_prob * stress_scale),
        delay_noise=base.delay_noise * (0.85 + 0.35 * stress_scale),
        burst_prob=min(0.95, base.burst_prob * stress_scale),
        burst_scale=base.burst_scale,
        robust_uncertainty_threshold=base.robust_uncertainty_threshold,
        robust_risk_lambda=base.robust_risk_lambda,
        robust_fallback_blend=base.robust_fallback_blend,
        robust_curriculum_max_scale=base.robust_curriculum_max_scale,
    )


def _evaluate_method(cfg: Config, method: str) -> tuple[float, float]:
    per_seed: list[float] = []
    for seed in cfg.seeds:
        if method == "heuristic_resilience":
            row = run_baseline(cfg, seed, method)
        elif method == "ppo_full":
            row = run_ppo(cfg, seed, robust=False)
        elif method == "robust_ppo":
            row = run_ppo(cfg, seed, robust=True)
        else:
            raise ValueError(method)
        per_seed.append(float(row["mean_reward_last5"]))

    mean = float(statistics.mean(per_seed))
    std = float(statistics.pstdev(per_seed)) if len(per_seed) > 1 else 0.0
    return mean, std


def main() -> int:
    base = Config()
    scales = [0.7, 1.0, 1.3, 1.6]
    methods = ["heuristic_resilience", "ppo_full", "robust_ppo"]

    records: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    for scale in scales:
        cfg = _scaled_cfg(base, scale)
        for method in methods:
            mean, std = _evaluate_method(cfg, method)
            records[method].append(
                {
                    "stress_scale": scale,
                    "attack_prob": cfg.attack_prob,
                    "delay_noise": cfg.delay_noise,
                    "burst_prob": cfg.burst_prob,
                    "mean_reward_last5": mean,
                    "std_reward_last5": std,
                }
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": {
            "attack_prob": base.attack_prob,
            "delay_noise": base.delay_noise,
            "burst_prob": base.burst_prob,
            "episodes": 34,
            "steps_per_episode": 24,
            "seeds": [3, 7, 11, 17],
        },
        "stress_scales": scales,
        "results": records,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = {
        "heuristic_resilience": "Heuristic Resilience",
        "ppo_full": "PPO",
        "robust_ppo": "Robust PPO (Safe)",
    }
    colors = {
        "heuristic_resilience": "#4C78A8",
        "ppo_full": "#E45756",
        "robust_ppo": "#54A24B",
    }

    plt.figure(figsize=(9.6, 5.0))
    for method in methods:
        xs = [r["stress_scale"] for r in records[method]]
        ys = [r["mean_reward_last5"] for r in records[method]]
        errs = [r["std_reward_last5"] for r in records[method]]
        plt.plot(xs, ys, marker="o", linewidth=2.0, color=colors[method], label=labels[method])
        lo = [y - e for y, e in zip(ys, errs)]
        hi = [y + e for y, e in zip(ys, errs)]
        plt.fill_between(xs, lo, hi, alpha=0.14, color=colors[method])

    plt.title("Robustness Sweep Across Adversarial Stress Levels")
    plt.xlabel("Stress Scale (attack + delay + burst multipliers)")
    plt.ylabel("Mean Reward (Last 5 Episodes)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=220)
    plt.close()

    print(f"Wrote stress sweep JSON: {OUT_JSON}")
    print(f"Wrote stress sweep figure: {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
