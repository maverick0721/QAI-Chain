from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai.models.vqc_policy import VQCPolicyNetwork, count_trainable_parameters
from ai.rl.scaled_environment import ScaledGovernanceEnv

ROOT = Path(__file__).resolve().parents[1]


class MLP28(nn.Module):
    """6->4->1 with no bias gives exactly 28 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 4, bias=False)
        self.fc2 = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


class ClassicalMLP(nn.Module):
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


class LoRAAdaptedMLP(nn.Module):
    """Frozen backbone with low-rank trainable adapters."""

    def __init__(self, hidden: int = 64, rank: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        for p in [*self.fc1.parameters(), *self.fc2.parameters(), *self.fc3.parameters()]:
            p.requires_grad = False

        self.a1 = nn.Linear(6, rank, bias=False)
        self.b1 = nn.Linear(rank, hidden, bias=False)
        self.a2 = nn.Linear(hidden, rank, bias=False)
        self.b2 = nn.Linear(rank, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.fc1(x) + self.b1(self.a1(x))
        h1 = torch.tanh(h1)
        h2 = self.fc2(h1) + self.b2(self.a2(h1))
        h2 = torch.tanh(h2)
        return self.fc3(h2)


class DistilledStudentMLP(nn.Module):
    """Small student-style MLP baseline."""

    def __init__(self, hidden: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _estimate_forward_flops(model: nn.Module) -> int:
    """Rough FLOPs estimate from trainable and frozen params.

    We use 2 FLOPs/parameter as a compact proxy for multiply-add in forward pass.
    """
    total_params = int(sum(p.numel() for p in model.parameters()))
    return int(2 * total_params)


def _evaluate_model(model_ctor, model_name: str, seeds: list[int], episodes: int, steps: int) -> dict:
    rewards: list[float] = []
    violations: list[float] = []
    episode_wall_clocks: list[float] = []
    per_step_latencies_ms: list[float] = []

    for seed in seeds:
        set_seed(seed)
        model = model_ctor()

        def act(state: np.ndarray) -> float:
            with torch.no_grad():
                if model_name == "vqc_6q_4l":
                    mean, _ = model(torch.tensor(state[:6], dtype=torch.float32))
                    return float(torch.tanh(mean.squeeze()).item())
                value = model(torch.tensor(state[:6], dtype=torch.float32))
                return float(torch.tanh(value.squeeze()).item())

        ep_rewards: list[float] = []
        ep_violations: list[float] = []
        for _ in range(episodes):
            env = ScaledGovernanceEnv()
            state = env.reset()
            ep_t0 = time.perf_counter()
            total_r = 0.0
            total_v = 0.0
            for _ in range(steps):
                step_t0 = time.perf_counter()
                a0 = act(state)
                per_step_latencies_ms.append((time.perf_counter() - step_t0) * 1000.0)
                action = np.asarray([a0, 0.25 * a0, 0.20 * a0], dtype=np.float32)
                state, reward, done, info = env.step(action, adversarial_intensity=0.7)
                total_r += float(reward)
                total_v += float(info["safety_violation"])
                if done:
                    break
            episode_wall_clocks.append(time.perf_counter() - ep_t0)
            ep_rewards.append(total_r)
            ep_violations.append(total_v)

        rewards.append(float(np.mean(ep_rewards)))
        violations.append(float(np.mean(ep_violations)))

    model_probe = model_ctor()
    params = count_trainable_parameters(model_probe)
    return {
        "params": int(params),
        "reward_mean": float(statistics.mean(rewards)),
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "violations_mean": float(statistics.mean(violations)),
        "violations_std": float(statistics.pstdev(violations)) if len(violations) > 1 else 0.0,
        "inference_latency_ms": float(statistics.mean(per_step_latencies_ms)) if per_step_latencies_ms else 0.0,
        "episode_wall_clock_s": float(statistics.mean(episode_wall_clocks)) if episode_wall_clocks else 0.0,
        "forward_flops_est": int(_estimate_forward_flops(model_probe)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    episodes = args.episodes
    steps = args.steps

    results = {
        "mlp_28": _evaluate_model(MLP28, "mlp_28", seeds, episodes, steps),
        "vqc_6q_4l": _evaluate_model(lambda: VQCPolicyNetwork(n_qubits=6, n_layers=4), "vqc_6q_4l", seeds, episodes, steps),
        "mlp_4673": _evaluate_model(lambda: ClassicalMLP(hidden=64), "mlp_4673", seeds, episodes, steps),
        "lora_mlp": _evaluate_model(lambda: LoRAAdaptedMLP(hidden=64, rank=2), "lora_mlp", seeds, episodes, steps),
        "distilled_student": _evaluate_model(lambda: DistilledStudentMLP(hidden=12), "distilled_student", seeds, episodes, steps),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seeds": seeds, "episodes": episodes, "steps": steps},
        "results": results,
    }

    out_json = ROOT / "experiments" / "results" / "parameter_efficiency_matched.json"
    out_md = ROOT / "docs" / "PARAMETER_EFFICIENCY_MATCHED.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    labels = {
        "mlp_28": "MLP (28 params)",
        "vqc_6q_4l": "VQC (6q, 4l)",
        "mlp_4673": "MLP (4673 params)",
        "lora_mlp": "LoRA-MLP (trainable)",
        "distilled_student": "Distilled Student",
    }
    for key in ["mlp_28", "vqc_6q_4l", "lora_mlp", "distilled_student", "mlp_4673"]:
        r = results[key]
        rows.append(
            f"| {labels[key]} | {r['params']} | {r['reward_mean']:.2f} ± {r['reward_std']:.2f} | {r['violations_mean']:.2f} ± {r['violations_std']:.2f} | {r['inference_latency_ms']:.4f} | {r['forward_flops_est']} |"
        )

    lines = [
        "# Matched-Capacity Parameter Efficiency",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "| Policy | Parameters | Mean Reward ± std | Safety Violations ± std | Inference ms/step | FLOPs (forward est.) |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = ROOT / "paper" / "tables" / "parameter_efficiency_matched.tex"
    table.write_text(
        "\n".join(
            [
                r"\footnotesize",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{lccccc}",
                r"\toprule",
                r"Policy & Parameters & Mean Reward & Safety Violations & Inference (ms) & FLOPs est. \\",
                r"\midrule",
                f"MLP (28 params) & {results['mlp_28']['params']} & {results['mlp_28']['reward_mean']:.2f} $\\pm$ {results['mlp_28']['reward_std']:.2f} & {results['mlp_28']['violations_mean']:.2f} $\\pm$ {results['mlp_28']['violations_std']:.2f} & {results['mlp_28']['inference_latency_ms']:.4f} & {results['mlp_28']['forward_flops_est']} \\\\",
                f"VQC (6q, 4l) & {results['vqc_6q_4l']['params']} & {results['vqc_6q_4l']['reward_mean']:.2f} $\\pm$ {results['vqc_6q_4l']['reward_std']:.2f} & {results['vqc_6q_4l']['violations_mean']:.2f} $\\pm$ {results['vqc_6q_4l']['violations_std']:.2f} & {results['vqc_6q_4l']['inference_latency_ms']:.4f} & {results['vqc_6q_4l']['forward_flops_est']} \\\\",
                f"LoRA-MLP (trainable) & {results['lora_mlp']['params']} & {results['lora_mlp']['reward_mean']:.2f} $\\pm$ {results['lora_mlp']['reward_std']:.2f} & {results['lora_mlp']['violations_mean']:.2f} $\\pm$ {results['lora_mlp']['violations_std']:.2f} & {results['lora_mlp']['inference_latency_ms']:.4f} & {results['lora_mlp']['forward_flops_est']} \\\\",
                f"Distilled Student & {results['distilled_student']['params']} & {results['distilled_student']['reward_mean']:.2f} $\\pm$ {results['distilled_student']['reward_std']:.2f} & {results['distilled_student']['violations_mean']:.2f} $\\pm$ {results['distilled_student']['violations_std']:.2f} & {results['distilled_student']['inference_latency_ms']:.4f} & {results['distilled_student']['forward_flops_est']} \\\\",
                f"MLP (4673 params) & {results['mlp_4673']['params']} & {results['mlp_4673']['reward_mean']:.2f} $\\pm$ {results['mlp_4673']['reward_std']:.2f} & {results['mlp_4673']['violations_mean']:.2f} $\\pm$ {results['mlp_4673']['violations_std']:.2f} & {results['mlp_4673']['inference_latency_ms']:.4f} & {results['mlp_4673']['forward_flops_est']} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
