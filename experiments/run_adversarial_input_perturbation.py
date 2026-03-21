from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai.governance.safety_shield import GovernanceSafetyShield, ShieldState
from ai.rl.scaled_environment import ScaledGovernanceEnv

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def heuristic_action(state: np.ndarray) -> float:
    # Difficulty is first state channel in ScaledGovernanceEnv.
    difficulty = float(state[0])
    target = 4.0
    return float(np.clip(2.5 * (target - difficulty), -8.0, 8.0))


def _run_mode(
    shielded: bool,
    noise_sigma: float,
    seeds: list[int],
    episodes: int,
    steps: int,
    adversarial_intensity: float,
) -> dict[str, float]:
    reward_means: list[float] = []
    viol_means: list[float] = []

    for seed in seeds:
        set_seed(seed)
        env = ScaledGovernanceEnv()
        shield = GovernanceSafetyShield() if shielded else None

        ep_rewards: list[float] = []
        ep_viols: list[float] = []
        for _ in range(episodes):
            state = env.reset()
            total_r = 0.0
            total_v = 0.0
            for step_idx in range(steps):
                noisy_state = state + np.random.normal(0.0, noise_sigma, size=state.shape).astype(np.float32)
                proposed = heuristic_action(noisy_state)
                if shield is not None:
                    decision_state = ShieldState(
                        difficulty=float(state[0]),
                        step=step_idx,
                        state_vector=state.tolist(),
                    )
                    _, executed = shield.validate_action(proposed, decision_state)
                else:
                    executed = proposed

                action = np.asarray([executed, 0.25 * executed, 0.20 * executed], dtype=np.float32)
                state, reward, done, info = env.step(action, adversarial_intensity=adversarial_intensity)
                total_r += float(reward)
                total_v += float(info["safety_violation"])
                if done:
                    break
            ep_rewards.append(total_r)
            ep_viols.append(total_v / float(steps))

        reward_means.append(float(np.mean(ep_rewards)))
        viol_means.append(float(np.mean(ep_viols)))

    return {
        "reward_mean": float(statistics.mean(reward_means)),
        "reward_std": float(statistics.pstdev(reward_means)) if len(reward_means) > 1 else 0.0,
        "violation_rate_mean": float(statistics.mean(viol_means)),
        "violation_rate_std": float(statistics.pstdev(viol_means)) if len(viol_means) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Adversarial input perturbation comparison: shielded vs unshielded.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sigmas", type=str, default="0.00,0.05,0.10,0.20")
    parser.add_argument("--adversarial-intensity", type=float, default=0.7)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]

    results: dict[str, dict[str, float]] = {}
    for sigma in sigmas:
        key = f"sigma_{sigma:.2f}"
        results[key] = {
            "shielded": _run_mode(True, sigma, seeds, args.episodes, args.steps, args.adversarial_intensity),
            "unshielded": _run_mode(False, sigma, seeds, args.episodes, args.steps, args.adversarial_intensity),
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seeds": seeds,
            "episodes": args.episodes,
            "steps": args.steps,
            "sigmas": sigmas,
            "adversarial_intensity": args.adversarial_intensity,
        },
        "results": results,
    }

    out_json = ROOT / "experiments" / "results" / "adversarial_input_perturbation.json"
    out_md = ROOT / "docs" / "ADVERSARIAL_INPUT_PERTURBATION.md"
    out_table = ROOT / "paper" / "tables" / "adversarial_input_perturbation.tex"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Adversarial Input Perturbation (Shielded vs Unshielded)",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "| Noise sigma | Controller | Reward mean±std | Violation-rate mean±std |",
        "|---:|---|---:|---:|",
    ]

    tex_lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Noise $\sigma$ & Controller & Reward & Violation Rate \\",
        r"\midrule",
    ]

    for sigma in sigmas:
        key = f"sigma_{sigma:.2f}"
        for ctrl in ["unshielded", "shielded"]:
            r = results[key][ctrl]
            ctrl_label = "Unshielded" if ctrl == "unshielded" else "Shielded"
            lines.append(
                f"| {sigma:.2f} | {ctrl_label} | {r['reward_mean']:.2f} ± {r['reward_std']:.2f} | {r['violation_rate_mean']:.4f} ± {r['violation_rate_std']:.4f} |"
            )
            tex_lines.append(
                f"{sigma:.2f} & {ctrl_label} & {r['reward_mean']:.2f} $\\pm$ {r['reward_std']:.2f} & {r['violation_rate_mean']:.4f} $\\pm$ {r['violation_rate_std']:.4f} \\\\"
            )

    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_table.write_text("\n".join(tex_lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
