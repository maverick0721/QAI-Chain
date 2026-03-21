from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai.rl.defi_environment import DeFiLiquidityGovernanceEnv
from ai.rl.scaled_environment import ScaledGovernanceEnv

ROOT = Path(__file__).resolve().parents[1]


def run_env(env_name: str, episodes: int = 30) -> dict[str, float]:
    rewards = {"heuristic": [], "random": [], "robust": []}

    for _ in range(episodes):
        if env_name == "scaled":
            env = ScaledGovernanceEnv()

            def heur(s):
                return np.asarray([np.clip(4.0 - s[0], -1.0, 1.0), np.clip(0.3 - s[4], -1.0, 1.0), np.clip(0.4 - s[6], -1.0, 1.0)], dtype=np.float32)

            def robust(s):
                h = heur(s)
                n = np.random.normal(0.0, 0.12, size=3)
                return np.clip(0.7 * (h + n) + 0.3 * h, -1.0, 1.0)

            steps = 180
        else:
            env = DeFiLiquidityGovernanceEnv()

            def heur(s):
                fee_delta = np.clip(0.003 - s[0], -1.0, 1.0)
                reb_delta = np.clip(0.12 - s[1], -1.0, 1.0)
                return np.asarray([fee_delta, reb_delta], dtype=np.float32)

            def robust(s):
                h = heur(s)
                n = np.random.normal(0.0, 0.10, size=2)
                return np.clip(0.6 * (h + n) + 0.4 * h, -1.0, 1.0)

            steps = 140

        for method in rewards.keys():
            state = env.reset()
            total = 0.0
            for _ in range(steps):
                if method == "heuristic":
                    action = heur(state)
                elif method == "robust":
                    action = robust(state)
                else:
                    action = np.random.uniform(-1.0, 1.0, size=env.cfg.action_dim).astype(np.float32)

                state, r, done, _ = env.step(action, adversarial_intensity=0.65) if env_name == "scaled" else env.step(action, adversarial_intensity=0.65)
                total += float(r)
                if done:
                    break
            rewards[method].append(total)

    return {k: float(np.mean(v)) for k, v in rewards.items()}


def main() -> int:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scaled_env": run_env("scaled", episodes=30),
        "defi_env": run_env("defi", episodes=30),
    }

    out_json = ROOT / "experiments" / "results" / "dual_environment_transfer.json"
    out_md = ROOT / "docs" / "DUAL_ENVIRONMENT_TRANSFER.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Dual-Environment Transfer", "", f"Generated (UTC): {payload['generated_at_utc']}", ""]
    lines.append("| Environment | Heuristic | Robust | Random |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Scaled Governance | {payload['scaled_env']['heuristic']:.2f} | {payload['scaled_env']['robust']:.2f} | {payload['scaled_env']['random']:.2f} |")
    lines.append(f"| DeFi Governance | {payload['defi_env']['heuristic']:.2f} | {payload['defi_env']['robust']:.2f} | {payload['defi_env']['random']:.2f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote transfer JSON: {out_json}")
    print(f"Wrote transfer markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
