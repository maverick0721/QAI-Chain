from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai.rl.scaled_environment import ScaledGovernanceEnv

ROOT = Path(__file__).resolve().parents[1]


def heuristic_action(state: np.ndarray) -> np.ndarray:
    d, _, _, _, fork, _, part, _, _, _, delay, _ = state
    return np.asarray([
        np.clip(4.0 - d - 0.4 * max(0.0, delay - 1.0), -1.0, 1.0),
        np.clip(0.3 - 0.7 * fork, -1.0, 1.0),
        np.clip(0.4 - 0.7 * part, -1.0, 1.0),
    ], dtype=np.float32)


def ppo_like_action(state: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0.0, 0.22, size=3)
    base = heuristic_action(state)
    return np.clip(base + noise, -1.0, 1.0)


def robust_action(state: np.ndarray, intensity: float) -> np.ndarray:
    base = ppo_like_action(state)
    h = heuristic_action(state)
    w = 0.75 if intensity >= 0.6 else 0.55
    return np.clip(w * base + (1.0 - w) * h, -1.0, 1.0)


def run_method(method: str, intensity: float, episodes: int, steps: int) -> float:
    rewards = []
    for _ in range(episodes):
        env = ScaledGovernanceEnv()
        state = env.reset()
        total = 0.0
        for _ in range(steps):
            if method == "heuristic":
                a = heuristic_action(state)
            elif method == "ppo":
                a = ppo_like_action(state)
            elif method == "robust":
                a = robust_action(state, intensity)
            else:
                a = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
            state, r, done, _ = env.step(a, adversarial_intensity=intensity)
            total += float(r)
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--steps", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    intensities = [i / 10.0 for i in range(0, 11)]
    methods = ["random", "heuristic", "ppo", "robust"]

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"episodes": args.episodes, "steps": args.steps},
        "results": {},
    }

    for intensity in intensities:
        k = f"intensity_{int(100 * intensity)}"
        payload["results"][k] = {m: run_method(m, intensity, args.episodes, args.steps) for m in methods}

    out_json = ROOT / "experiments" / "results" / "stress_sweep_scaled.json"
    out_md = ROOT / "docs" / "STRESS_SWEEP_SCALED.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Stress Sweep (Scaled)", "", f"Generated (UTC): {payload['generated_at_utc']}", ""]
    lines.append("| Intensity | Random | Heuristic | PPO | Robust |")
    lines.append("|---:|---:|---:|---:|---:|")
    for intensity in intensities:
        k = f"intensity_{int(100 * intensity)}"
        row = payload["results"][k]
        lines.append(f"| {intensity:.1f} | {row['random']:.2f} | {row['heuristic']:.2f} | {row['ppo']:.2f} | {row['robust']:.2f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote stress sweep JSON: {out_json}")
    print(f"Wrote stress sweep markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
