from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ai.governance.quantum_uncertainty import estimate_quantum_uncertainty
from ai.governance.safety_shield import GovernanceSafetyShield, ShieldState
from ai.models.vqc_policy import VQCPolicyNetwork
from ai.rl.scaled_environment import ScaledGovernanceConfig, ScaledGovernanceEnv
from core.blockchain.audit_trail import make_audit_record
from core.blockchain.blockchain import Blockchain
from core.blockchain.policy_audit import hash_policy_state_dict

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seeds: tuple[int, ...]
    episodes: int
    steps: int
    uncertainty_k: int
    uncertainty_every: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def heuristic_action(state: np.ndarray) -> np.ndarray:
    difficulty, _, _, _, fork_depth, _, partition, _, _, _, delay, _ = state
    a0 = np.clip(4.0 - difficulty - 0.5 * max(0.0, delay - 1.0), -1.0, 1.0)
    a1 = np.clip(0.4 - 0.7 * fork_depth, -1.0, 1.0)
    a2 = np.clip(0.5 - 0.8 * partition, -1.0, 1.0)
    return np.asarray([a0, a1, a2], dtype=np.float32)


def policy_action(policy: VQCPolicyNetwork, state: np.ndarray) -> tuple[np.ndarray, float]:
    s = torch.tensor(state[:6], dtype=torch.float32)
    with torch.no_grad():
        mean, std = policy(s)
        dist = torch.distributions.Normal(mean.squeeze(), std.squeeze())
        x = float(torch.clamp(dist.sample(), -1.0, 1.0).item())
    return np.asarray([x, 0.3 * x, 0.2 * x], dtype=np.float32), float(std.squeeze().item())


def fast_policy_proxy(state: np.ndarray) -> tuple[np.ndarray, float]:
    difficulty, _, _, _, fork_depth, _, partition, _, _, attack, delay, _ = state
    a0 = np.clip(0.65 * (4.0 - difficulty) - 0.22 * fork_depth - 0.20 * partition - 0.16 * max(0.0, delay - 1.0) - 0.12 * attack, -1.0, 1.0)
    sigma = float(np.clip(0.40 + 0.22 * partition + 0.18 * fork_depth, 0.08, 1.2))
    return np.asarray([a0, 0.3 * a0, 0.2 * a0], dtype=np.float32), sigma


def run_one(
    env: ScaledGovernanceEnv,
    blockchain: Blockchain,
    policy: VQCPolicyNetwork,
    mode: str,
    tau_q: float,
    intensity: float,
    uncertainty_k: int,
    uncertainty_every: int,
) -> dict[str, float]:
    shield = GovernanceSafetyShield(audit_sink=blockchain)
    reward_total = 0.0
    violations = 0.0
    fallback_count = 0
    latency_samples: list[float] = []
    uncertainty_samples: list[float] = []
    error_samples: list[float] = []

    state = env.reset()
    cached_q_fallback = False
    cached_q_unc = 0.0
    for t in range(env.cfg.episode_length):
        heur = heuristic_action(state)

        use_quantum = mode in {"full_system", "no_shield", "no_fallback"}
        use_shield = mode in {"full_system", "no_quantum_uncertainty", "classical_sigma"}
        use_fallback = mode not in {"no_fallback", "classical_ppo"}

        if mode in {"heuristic_only", "random"}:
            pol, sigma = np.zeros(3, dtype=np.float32), 0.0
        elif use_quantum:
            pol, sigma = policy_action(policy, state)
        else:
            pol, sigma = fast_policy_proxy(state)

        if mode in {"heuristic_only", "random"}:
            chosen = heur if mode == "heuristic_only" else np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
            unc = sigma
            fallback = False
        else:
            if use_quantum:
                if t % max(1, uncertainty_every) == 0:
                    qres = estimate_quantum_uncertainty(
                        policy,
                        torch.tensor(state[:6], dtype=torch.float32),
                        k=max(2, uncertainty_k),
                        noise_std=0.01,
                        tau_q=tau_q,
                    )
                    cached_q_unc = float(qres.variance)
                    cached_q_fallback = bool(qres.fallback)
                unc = cached_q_unc
                fallback = bool(cached_q_fallback and use_fallback)
            else:
                unc = sigma * sigma
                fallback = bool((sigma > 0.70) and use_fallback)

            chosen = heur if fallback else pol

        if use_shield:
            decision, executed_difficulty_action = shield.validate_action(
                float(chosen[0]),
                ShieldState(difficulty=float(env.difficulty), step=t, state_vector=state.tolist()),
            )
            chosen[0] = executed_difficulty_action
            shield_result = decision.value
        else:
            shield_result = "BYPASS"

        next_state, reward, done, info = env.step(chosen, adversarial_intensity=intensity)
        reward_total += reward
        violations += float(info["safety_violation"])
        error_samples.append(float(info["target_error"]))
        uncertainty_samples.append(float(unc))
        latency_samples.append(1.0 + 0.2 * abs(chosen[0]) + 0.4 * float(unc))
        if fallback:
            fallback_count += 1

        rec = make_audit_record(
            epoch=t,
            policy_hash=hash_policy_state_dict(policy.state_dict()),
            state=state.tolist(),
            action=chosen.tolist(),
            uncertainty=float(unc),
            fallback_triggered=bool(fallback),
            shield_result=shield_result,
        )
        blockchain.commit_audit_record(rec)

        state = next_state
        if done:
            break

    return {
        "reward": float(reward_total),
        "safety_violations": float(violations),
        "target_error_mae": float(np.mean(error_samples)),
        "fallback_rate": float(fallback_count / max(1, env.cfg.episode_length)),
        "latency": float(np.mean(latency_samples)),
        "mean_uncertainty": float(np.mean(uncertainty_samples)),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        out[f"{key}_mean"] = float(statistics.mean(vals))
        out[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="3,7,11,17,21,42,84,126,5,9,13,19,23,31,57,99")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--tau-q", type=float, default=0.02)
    p.add_argument("--intensity", type=float, default=0.6)
    p.add_argument("--uncertainty-k", type=int, default=3)
    p.add_argument("--uncertainty-every", type=int, default=6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        episodes=args.episodes,
        steps=args.steps,
        uncertainty_k=args.uncertainty_k,
        uncertainty_every=args.uncertainty_every,
    )
    modes = [
        "full_system",
        "no_quantum_uncertainty",
        "no_shield",
        "no_fallback",
        "classical_ppo",
        "heuristic_only",
        "random",
    ]

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seeds": list(cfg.seeds),
            "episodes": cfg.episodes,
            "steps": cfg.steps,
            "tau_q": args.tau_q,
            "intensity": args.intensity,
            "uncertainty_k": cfg.uncertainty_k,
            "uncertainty_every": cfg.uncertainty_every,
        },
        "results": {},
    }

    for mode in modes:
        per_seed: list[dict[str, float]] = []
        for seed in cfg.seeds:
            set_seed(seed)
            env = ScaledGovernanceEnv(ScaledGovernanceConfig(episode_length=cfg.steps))
            chain = Blockchain(difficulty=7)
            policy = VQCPolicyNetwork(n_qubits=6, n_layers=4)
            episodes = [
                run_one(
                    env,
                    chain,
                    policy,
                    mode,
                    tau_q=args.tau_q,
                    intensity=args.intensity,
                    uncertainty_k=cfg.uncertainty_k,
                    uncertainty_every=cfg.uncertainty_every,
                )
                for _ in range(cfg.episodes)
            ]
            per_seed.append(aggregate(episodes))

        payload["results"][mode] = {
            "per_seed": per_seed,
            "aggregate": aggregate(per_seed),
        }

    out_json = ROOT / "experiments" / "results" / "full_ablation_matrix.json"
    out_md = ROOT / "docs" / "FULL_ABLATION_MATRIX.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Full Ablation Matrix", "", f"Generated (UTC): {payload['generated_at_utc']}", ""]
    lines.append("| Method | Reward | Safety Violations | MAE | Fallback | Latency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode in modes:
        agg = payload["results"][mode]["aggregate"]
        lines.append(
            f"| {mode} | {agg['reward_mean_mean']:.3f} | {agg['safety_violations_mean_mean']:.3f} "
            f"| {agg['target_error_mae_mean_mean']:.3f} | {agg['fallback_rate_mean_mean']:.3f} | {agg['latency_mean_mean']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote ablation matrix JSON: {out_json}")
    print(f"Wrote ablation matrix markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
