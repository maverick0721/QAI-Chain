from __future__ import annotations

import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai.models.vqc_policy import VQCPolicyNetwork, count_trainable_parameters
from ai.rl.scaled_environment import ScaledGovernanceEnv

ROOT = Path(__file__).resolve().parents[1]


class ClassicalPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(model_name: str, seeds: list[int], episodes: int, steps: int):
    rewards = []
    violations = []

    for seed in seeds:
        set_seed(seed)
        if model_name == "classical_mlp":
            model = ClassicalPolicy(hidden=64)

            def act(s):
                with torch.no_grad():
                    v = model(torch.tensor(s, dtype=torch.float32))
                    return float(torch.tanh(v).item())

        else:
            model = VQCPolicyNetwork(n_qubits=6, n_layers=4)

            def act(s):
                with torch.no_grad():
                    mean, _ = model(torch.tensor(s, dtype=torch.float32))
                    return float(torch.tanh(mean.squeeze()).item())

        total_rewards = []
        total_viol = []
        for _ in range(episodes):
            env = ScaledGovernanceEnv()
            state = env.reset()
            ep_reward = 0.0
            ep_viol = 0.0
            for _ in range(steps):
                a0 = act(state[:6])
                action = np.asarray([a0, 0.25 * a0, 0.20 * a0], dtype=np.float32)
                state, r, done, info = env.step(action, adversarial_intensity=0.7)
                ep_reward += float(r)
                ep_viol += float(info["safety_violation"])
                if done:
                    break
            total_rewards.append(ep_reward)
            total_viol.append(ep_viol)

        rewards.append(float(np.mean(total_rewards)))
        violations.append(float(np.mean(total_viol)))

    params = count_trainable_parameters(ClassicalPolicy(hidden=64)) if model_name == "classical_mlp" else count_trainable_parameters(VQCPolicyNetwork(n_qubits=6, n_layers=4))

    return {
        "params": int(params),
        "reward_mean": float(statistics.mean(rewards)),
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "violations_mean": float(statistics.mean(violations)),
        "violations_std": float(statistics.pstdev(violations)) if len(violations) > 1 else 0.0,
    }


def main() -> int:
    seeds = [3, 5, 7, 9, 11, 13, 17, 19, 21, 23, 27, 31, 37, 41, 42, 47, 53, 57, 63, 69, 73, 79, 84, 89, 97, 99, 107, 113, 127, 131]
    episodes = 10
    steps = 120

    classical = evaluate_model("classical_mlp", seeds, episodes, steps)
    vqc = evaluate_model("vqc_6q_4l", seeds, episodes, steps)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seeds": seeds, "episodes": episodes, "steps": steps},
        "results": {
            "classical_mlp": classical,
            "vqc_6q_4l": vqc,
        },
        "vqc_parameter_ratio_vs_classical": float(classical["params"] / max(1, vqc["params"])),
    }

    out_json = ROOT / "experiments" / "results" / "parameter_efficiency.json"
    out_md = ROOT / "docs" / "PARAMETER_EFFICIENCY.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Parameter Efficiency", "", f"Generated (UTC): {payload['generated_at_utc']}", ""]
    lines.append("| Policy | Parameters | Mean Reward ± std | Safety Violations |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Classical MLP | {classical['params']} | {classical['reward_mean']:.2f} ± {classical['reward_std']:.2f} | {classical['violations_mean']:.2f} |"
    )
    lines.append(
        f"| VQC (6-qubit, 4-layer) | {vqc['params']} | {vqc['reward_mean']:.2f} ± {vqc['reward_std']:.2f} | {vqc['violations_mean']:.2f} |"
    )
    lines.append("")
    lines.append(f"Parameter ratio (classical / VQC): {payload['vqc_parameter_ratio_vs_classical']:.2f}x")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote parameter efficiency JSON: {out_json}")
    print(f"Wrote parameter efficiency markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
