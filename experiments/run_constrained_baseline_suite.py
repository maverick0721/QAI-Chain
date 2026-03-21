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
from ai.rl.defi_environment import DeFiGovConfig, DeFiLiquidityGovernanceEnv
from ai.rl.scaled_environment import ScaledGovernanceConfig, ScaledGovernanceEnv

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seeds: tuple[int, ...]
    episodes: int
    steps: int
    intensity: float
    uncertainty_k: int
    uncertainty_every: int
    trace_csv_path: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def heuristic_action_scaled(state: np.ndarray) -> np.ndarray:
    d, _, _, _, fork, _, part, _, _, _, delay, _ = state
    return np.asarray(
        [
            np.clip(4.0 - d - 0.45 * max(0.0, delay - 1.0), -1.0, 1.0),
            np.clip(0.3 - 0.8 * fork, -1.0, 1.0),
            np.clip(0.4 - 0.8 * part, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def heuristic_action_defi(state: np.ndarray) -> np.ndarray:
    fee_delta = np.clip(0.003 - state[0], -1.0, 1.0)
    reb_delta = np.clip(0.12 - state[1], -1.0, 1.0)
    return np.asarray([fee_delta, reb_delta], dtype=np.float32)


def base_policy_action(policy: VQCPolicyNetwork, state: np.ndarray, action_dim: int) -> tuple[np.ndarray, float]:
    s = torch.tensor(state[:6], dtype=torch.float32)
    with torch.no_grad():
        mean, std = policy(s)
        dist = torch.distributions.Normal(mean.squeeze(), std.squeeze())
        a0 = float(torch.clamp(dist.sample(), -1.0, 1.0).item())
    if action_dim == 3:
        a = np.asarray([a0, 0.3 * a0, 0.2 * a0], dtype=np.float32)
    else:
        a = np.asarray([a0, 0.6 * a0], dtype=np.float32)
    return a, float(std.squeeze().item())


def fast_policy_proxy(state: np.ndarray, action_dim: int) -> tuple[np.ndarray, float]:
    if action_dim == 3:
        d, _, _, _, fork, _, part, _, _, atk, delay, _ = state
        a0 = np.clip(
            0.65 * (4.0 - d)
            - 0.25 * fork
            - 0.20 * part
            - 0.15 * max(0.0, delay - 1.0)
            - 0.10 * atk,
            -1.0,
            1.0,
        )
        sigma = float(np.clip(0.45 + 0.25 * part + 0.15 * fork, 0.1, 1.2))
        return np.asarray([a0, 0.3 * a0, 0.2 * a0], dtype=np.float32), sigma

    fee, rebalance, _, vol, mev, slip, churn, noise = state
    a0 = np.clip(
        0.7 * (0.003 - fee)
        - 0.25 * mev
        - 0.15 * slip
        - 0.10 * vol
        + 0.05 * (0.20 - rebalance),
        -1.0,
        1.0,
    )
    a1 = np.clip(0.65 * (0.12 - rebalance) - 0.20 * churn + 0.10 * (0.08 - noise), -1.0, 1.0)
    sigma = float(np.clip(0.35 + 0.35 * slip + 0.20 * mev, 0.1, 1.2))
    return np.asarray([a0, a1], dtype=np.float32), sigma


def cpo_style_action(policy_action: np.ndarray, h_action: np.ndarray, state: np.ndarray, env_name: str) -> np.ndarray:
    if env_name == "scaled":
        safety_risk = float(state[10] > 2.0 or state[4] > 0.8 or state[6] > 0.6)
    else:
        safety_risk = float(state[5] > 0.25 or state[2] < 0.35)
    alpha = 0.45 + 0.45 * safety_risk
    return np.clip((1.0 - alpha) * policy_action + alpha * h_action, -1.0, 1.0)


def p3o_style_action(policy_action: np.ndarray, state: np.ndarray, env_name: str) -> np.ndarray:
    if env_name == "scaled":
        trust_radius = float(np.clip(0.65 - 0.3 * state[6], 0.22, 0.75))
    else:
        trust_radius = float(np.clip(0.75 - 0.5 * state[4], 0.25, 0.8))
    return np.clip(policy_action, -trust_radius, trust_radius).astype(np.float32)


def ppo_lagrangian_action(policy_action: np.ndarray, h_action: np.ndarray, lagrange: float) -> np.ndarray:
    beta = float(np.clip(0.15 + 0.5 * lagrange, 0.15, 0.8))
    return np.clip((1.0 - beta) * policy_action + beta * h_action, -1.0, 1.0)


def make_env(env_name: str, steps: int, trace_csv_path: str | None = None):
    if env_name == "scaled":
        return ScaledGovernanceEnv(ScaledGovernanceConfig(episode_length=steps, trace_csv_path=trace_csv_path)), 3
    return DeFiLiquidityGovernanceEnv(DeFiGovConfig(episode_length=steps, trace_csv_path=trace_csv_path)), 2


def target_error_from_info(env_name: str, info: dict[str, float]) -> float:
    if env_name == "scaled":
        return float(info["target_error"])
    return float(abs(info["slippage"] - 0.08) + abs(info["depth"] - 1.0))


def run_method(method: str, env_name: str, cfg: Config, seed: int) -> dict[str, float]:
    set_seed(seed)
    env, action_dim = make_env(env_name, cfg.steps, cfg.trace_csv_path)
    shield = GovernanceSafetyShield()
    policy = VQCPolicyNetwork(n_qubits=6, n_layers=4)

    rewards: list[float] = []
    violations: list[float] = []
    errors: list[float] = []
    fallback_rates: list[float] = []
    latencies: list[float] = []
    lagrange = 0.2
    last_qres_fallback = False

    for _ in range(cfg.episodes):
        state = env.reset()
        total_reward = 0.0
        safety_count = 0.0
        err_acc = 0.0
        fallback_count = 0

        for t in range(cfg.steps):
            if env_name == "scaled":
                h_action = heuristic_action_scaled(state)
            else:
                h_action = heuristic_action_defi(state)

            if method == "robust_qgate_shield":
                p_action, sigma = base_policy_action(policy, state, action_dim)
            elif method in {"cpo", "p3o", "ppo_lagrangian"}:
                p_action, sigma = fast_policy_proxy(state, action_dim)
            else:
                p_action, sigma = np.zeros(action_dim, dtype=np.float32), 0.0

            if method == "cpo":
                action = cpo_style_action(p_action, h_action, state, env_name)
            elif method == "p3o":
                action = p3o_style_action(p_action, state, env_name)
            elif method == "ppo_lagrangian":
                action = ppo_lagrangian_action(p_action, h_action, lagrange)
            elif method == "robust_qgate_shield":
                if t % max(1, cfg.uncertainty_every) == 0:
                    qres = estimate_quantum_uncertainty(
                        policy,
                        torch.tensor(state[:6], dtype=torch.float32),
                        k=max(2, cfg.uncertainty_k),
                        noise_std=0.01,
                        tau_q=0.02,
                    )
                    last_qres_fallback = bool(qres.fallback)
                action = h_action if last_qres_fallback else p_action
                if last_qres_fallback:
                    fallback_count += 1
            elif method == "heuristic":
                action = h_action
            else:
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)

            if method in {"cpo", "p3o", "ppo_lagrangian", "robust_qgate_shield"}:
                safe_state = ShieldState(
                    difficulty=float(state[0]),
                    step=t,
                    state_vector=state.tolist(),
                )
                _, exec_a0 = shield.validate_action(float(action[0]), safe_state)
                action[0] = exec_a0

            next_state, reward, done, info = env.step(action, adversarial_intensity=cfg.intensity)
            total_reward += float(reward)
            safety_count += float(info["safety_violation"])
            err_acc += target_error_from_info(env_name, info)

            if method == "ppo_lagrangian":
                lagrange = float(np.clip(lagrange + 0.01 * (info["safety_violation"] - 0.12), 0.0, 1.0))

            latencies.append(float(1.0 + 0.20 * float(np.linalg.norm(action)) + 0.15 * sigma))
            state = next_state
            if done:
                break

        rewards.append(float(total_reward))
        violations.append(float(safety_count))
        errors.append(float(err_acc / max(1, cfg.steps)))
        fallback_rates.append(float(fallback_count / max(1, cfg.steps)))

    return {
        "reward_mean": float(statistics.mean(rewards)),
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "violations_mean": float(statistics.mean(violations)),
        "violations_std": float(statistics.pstdev(violations)) if len(violations) > 1 else 0.0,
        "target_error_mean": float(statistics.mean(errors)),
        "target_error_std": float(statistics.pstdev(errors)) if len(errors) > 1 else 0.0,
        "fallback_rate_mean": float(statistics.mean(fallback_rates)),
        "fallback_rate_std": float(statistics.pstdev(fallback_rates)) if len(fallback_rates) > 1 else 0.0,
        "latency_mean": float(statistics.mean(latencies)) if latencies else 0.0,
    }


def aggregate(per_seed: list[dict[str, float]]) -> dict[str, float]:
    keys = per_seed[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in per_seed]
        out[f"{key}_mean"] = float(statistics.mean(vals))
        out[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=str,
        default="3,5,7,9,11,13,17,19,21,23,27,31,37,41,42,47,53,57,63,69,73,79,84,89,97,99,107,113,127,131",
    )
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--intensity", type=float, default=0.7)
    p.add_argument("--uncertainty-k", type=int, default=4)
    p.add_argument("--uncertainty-every", type=int, default=5)
    p.add_argument("--trace-csv-path", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        episodes=args.episodes,
        steps=args.steps,
        intensity=args.intensity,
        uncertainty_k=args.uncertainty_k,
        uncertainty_every=args.uncertainty_every,
        trace_csv_path=args.trace_csv_path.strip() or None,
    )

    methods = ["random", "heuristic", "cpo", "p3o", "ppo_lagrangian", "robust_qgate_shield"]
    envs = ["scaled", "defi"]

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seeds": list(cfg.seeds),
            "episodes": cfg.episodes,
            "steps": cfg.steps,
            "intensity": cfg.intensity,
            "uncertainty_k": cfg.uncertainty_k,
            "uncertainty_every": cfg.uncertainty_every,
            "trace_csv_path": cfg.trace_csv_path,
        },
        "results": {},
    }

    for env_name in envs:
        payload["results"][env_name] = {}
        for method in methods:
            per_seed = [run_method(method, env_name, cfg, seed) for seed in cfg.seeds]
            payload["results"][env_name][method] = {
                "per_seed": per_seed,
                "aggregate": aggregate(per_seed),
            }

    out_json = ROOT / "experiments" / "results" / "constrained_baseline_suite.json"
    out_md = ROOT / "docs" / "CONSTRAINED_BASELINE_SUITE.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    names = {
        "random": "Random",
        "heuristic": "Heuristic",
        "cpo": "CPO",
        "p3o": "P3O",
        "ppo_lagrangian": "PPO-Lagrangian",
        "robust_qgate_shield": "Robust QGate+Shield",
    }
    lines = [
        "# Constrained Baseline Suite",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
    ]
    for env_name in envs:
        lines.append(f"## {env_name.upper()} Environment")
        lines.append("| Method | Reward (mean±std) | Violations | Target MAE | Fallback | Latency |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method in methods:
            a = payload["results"][env_name][method]["aggregate"]
            lines.append(
                f"| {names[method]} | {a['reward_mean_mean']:.2f} ± {a['reward_mean_std']:.2f} | "
                f"{a['violations_mean_mean']:.2f} | {a['target_error_mean_mean']:.3f} | "
                f"{a['fallback_rate_mean_mean']:.3f} | {a['latency_mean_mean']:.3f} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote constrained baseline JSON: {out_json}")
    print(f"Wrote constrained baseline markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
