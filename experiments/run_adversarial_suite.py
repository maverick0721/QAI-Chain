# mypy: ignore-errors

from __future__ import annotations

import json
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.models.metrics_encoder import MetricsEncoder
from ai.models.policy_network import PolicyNetwork
from ai.models.value_network import ValueNetwork
from ai.rl.ppo_agent import PPOAgent


@dataclass
class Config:
    episodes: int = 55
    steps_per_episode: int = 30
    seeds: tuple[int, ...] = (3, 7, 11, 17, 21, 42, 84, 126, 5, 9, 13, 19, 23, 31, 57, 99)
    start_difficulty: int = 7
    target_difficulty: int = 3
    min_difficulty: int = 1
    max_difficulty: int = 12
    attack_prob: float = 0.28
    delay_noise: float = 0.55
    burst_prob: float = 0.18
    burst_scale: float = 2.3
    robust_uncertainty_threshold: float = 0.705
    robust_risk_lambda: float = 0.18
    robust_fallback_blend: float = 1.0
    robust_curriculum_max_scale: float = 1.0


class AdversarialGovernanceEnv:
    """A harder governance environment with stochastic attacks and delay bursts."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.target_difficulty = cfg.target_difficulty
        self.difficulty = float(cfg.start_difficulty)

        self.current_attack_prob = cfg.attack_prob
        self.current_delay_noise = cfg.delay_noise
        self.current_burst_prob = cfg.burst_prob

        self.attack_level = 0.0
        self.delay = 1.0
        self.backlog = 0.0
        self.fork_pressure = 0.0
        self.throughput = 0.0

        self.state_dim = 5

    def reset(self) -> np.ndarray:
        self.difficulty = float(self.cfg.start_difficulty)
        self.attack_level = 0.0
        self.delay = 1.0
        self.backlog = 0.0
        self.fork_pressure = 0.0
        self.throughput = 0.0
        return self.state()

    def configure_stress(self, scale: float) -> None:
        scale = float(max(0.1, scale))
        self.current_attack_prob = float(np.clip(self.cfg.attack_prob * scale, 0.0, 0.95))
        self.current_delay_noise = float(max(0.05, self.cfg.delay_noise * scale))
        self.current_burst_prob = float(np.clip(self.cfg.burst_prob * scale, 0.0, 0.95))

    def state(self) -> np.ndarray:
        return np.array(
            [
                self.difficulty,
                self.delay,
                self.backlog,
                self.attack_level,
                self.fork_pressure,
            ],
            dtype=np.float32,
        )

    def _sample_demand(self) -> float:
        base = np.random.normal(loc=6.0, scale=1.1)
        burst = self.cfg.burst_scale if np.random.rand() < self.current_burst_prob else 0.0
        return max(0.0, float(base + burst))

    def _sample_attack(self) -> float:
        if np.random.rand() < self.current_attack_prob:
            return float(np.random.uniform(0.5, 1.6))
        return float(np.random.uniform(0.0, 0.3))

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        delta = float(np.clip(action, -1.0, 1.0))
        self.difficulty = float(
            np.clip(
                self.difficulty + delta,
                self.cfg.min_difficulty,
                self.cfg.max_difficulty,
            )
        )

        demand = self._sample_demand()
        self.attack_level = self._sample_attack()

        service = max(0.2, 7.5 / (self.difficulty + 0.7))
        attack_penalty = 1.0 + 0.7 * self.attack_level
        serviced = max(0.0, service / attack_penalty)

        self.backlog = max(0.0, self.backlog + demand - serviced)
        self.delay = 1.0 + 0.18 * self.backlog + np.random.normal(0.0, self.current_delay_noise)
        self.delay = float(max(0.2, self.delay))

        self.fork_pressure = float(
            max(
                0.0,
                0.05 * self.attack_level + 0.035 * max(0.0, self.delay - 1.0),
            )
        )
        self.throughput = serviced

        # Harder multi-objective reward under adversarial conditions.
        target_error = abs(self.difficulty - self.target_difficulty)
        reward = (
            1.2 * self.throughput
            - 0.9 * self.delay
            - 1.0 * self.attack_level
            - 1.3 * self.fork_pressure
            - 0.45 * target_error
        )

        info = {
            "demand": float(demand),
            "serviced": float(serviced),
            "attack_level": float(self.attack_level),
            "delay": float(self.delay),
            "backlog": float(self.backlog),
            "fork_pressure": float(self.fork_pressure),
            "difficulty": float(self.difficulty),
            "target_error": float(target_error),
        }
        done = False
        return self.state(), float(reward), done, info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize_episode_stats(episodes: list[dict[str, float]]) -> dict[str, float]:
    last5 = episodes[-5:] if len(episodes) >= 5 else episodes

    def m(key: str, xs: list[dict[str, float]]) -> float:
        return float(statistics.mean(x[key] for x in xs)) if xs else 0.0

    return {
        "mean_reward_last5": m("reward", last5),
        "mean_reward_all": m("reward", episodes),
        "mean_delay_last5": m("delay", last5),
        "mean_attack_last5": m("attack", last5),
        "mean_target_error_last5": m("target_error", last5),
        "mean_fork_pressure_last5": m("fork_pressure", last5),
        "mean_fallback_rate_last5": m("fallback_rate", last5),
        "mean_uncertainty_last5": m("uncertainty", last5),
    }


def heuristic_resilience_action(env: AdversarialGovernanceEnv) -> float:
    target_pull = env.target_difficulty - env.difficulty
    resilience_push = -0.45 * max(0.0, env.delay - 1.4) - 0.55 * env.attack_level
    return float(np.clip(0.85 * target_pull + resilience_push, -1.0, 1.0))


def conservative_reward_estimate(env: AdversarialGovernanceEnv, action: float) -> float:
    d_next = float(
        np.clip(
            env.difficulty + float(np.clip(action, -1.0, 1.0)),
            env.cfg.min_difficulty,
            env.cfg.max_difficulty,
        )
    )

    expected_demand = 6.0 + env.current_burst_prob * env.cfg.burst_scale
    expected_attack = env.current_attack_prob * 1.05 + (1.0 - env.current_attack_prob) * 0.15

    service = max(0.2, 7.5 / (d_next + 0.7))
    attack_penalty = 1.0 + 0.7 * expected_attack
    serviced = max(0.0, service / attack_penalty)

    backlog_next = max(0.0, env.backlog + expected_demand - serviced)
    delay_next = max(0.2, 1.0 + 0.18 * backlog_next)
    fork_next = max(0.0, 0.05 * expected_attack + 0.035 * max(0.0, delay_next - 1.0))
    target_error = abs(d_next - env.target_difficulty)

    return float(
        1.2 * serviced
        - 0.9 * delay_next
        - 1.0 * expected_attack
        - 1.3 * fork_next
        - 0.45 * target_error
    )


def run_baseline(cfg: Config, seed: int, method: str) -> dict[str, float]:
    set_seed(seed)
    env = AdversarialGovernanceEnv(cfg)

    episodes: list[dict[str, float]] = []
    for _ in range(cfg.episodes):
        _ = env.reset()
        total_reward = 0.0
        delays: list[float] = []
        attacks: list[float] = []
        target_errors: list[float] = []
        fork_pressures: list[float] = []

        for _ in range(cfg.steps_per_episode):
            if method == "random":
                action = float(np.random.uniform(-1.0, 1.0))
            elif method == "heuristic_target":
                action = float(np.clip(env.target_difficulty - env.difficulty, -1.0, 1.0))
            elif method == "heuristic_resilience":
                action = heuristic_resilience_action(env)
            else:
                raise ValueError(method)

            _, reward, _, info = env.step(action)
            total_reward += reward
            delays.append(info["delay"])
            attacks.append(info["attack_level"])
            target_errors.append(info["target_error"])
            fork_pressures.append(info["fork_pressure"])

        episodes.append(
            {
                "reward": float(total_reward),
                "delay": float(statistics.mean(delays)),
                "attack": float(statistics.mean(attacks)),
                "target_error": float(statistics.mean(target_errors)),
                "fork_pressure": float(statistics.mean(fork_pressures)),
                "fallback_rate": 0.0,
                "uncertainty": 0.0,
            }
        )

    return summarize_episode_stats(episodes)


def run_ppo(
    cfg: Config,
    seed: int,
    robust: bool = False,
    mode: str | None = None,
) -> dict[str, float]:
    set_seed(seed)
    env = AdversarialGovernanceEnv(cfg)

    mode_name = mode or ("robust_ppo" if robust else "ppo_full")
    use_risk = mode_name in {"robust_ppo", "robust_risk_only"}
    use_fallback = mode_name in {"robust_ppo", "robust_fallback_only"}
    use_shield = mode_name == "robust_ppo"

    encoder = MetricsEncoder()
    policy = PolicyNetwork()
    value = ValueNetwork()
    agent = PPOAgent(policy, value)

    if mode_name != "ppo_full":
        # Use lower exploration noise for robust variants to improve calibration.
        with torch.no_grad():
            policy.log_std.fill_(-0.35)

    episodes: list[dict[str, float]] = []

    for ep in range(cfg.episodes):
        progress = ep / max(1, cfg.episodes - 1)
        agent.entropy_coef = 0.012 + (0.003 - 0.012) * progress

        env.configure_stress(1.0)

        state = env.reset()
        states: list[np.ndarray] = []
        actions: list[float] = []
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        raw_rewards: list[float] = []
        delays: list[float] = []
        attacks: list[float] = []
        target_errors: list[float] = []
        fork_pressures: list[float] = []
        fallback_steps = 0
        uncertainties: list[float] = []
        rolling_raw_rewards: list[float] = []

        for _ in range(cfg.steps_per_episode):
            encoded = encoder(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            encoded_np = encoded.detach().numpy()[0]

            if use_fallback:
                encoded_t = torch.as_tensor(encoded_np, dtype=torch.float32).unsqueeze(0)
                mean, std = policy(encoded_t)
                dist = torch.distributions.Normal(mean, std)

                ppo_action = float(dist.sample().item())
                uncertainty = float(std.mean().item())
                uncertainties.append(uncertainty)

                if uncertainty > cfg.robust_uncertainty_threshold:
                    fallback_steps += 1
                    fallback_action = heuristic_resilience_action(env)
                    blend = cfg.robust_fallback_blend if mode_name == "robust_ppo" else 0.7
                    action = float(
                        np.clip((1.0 - blend) * ppo_action + blend * fallback_action, -1.0, 1.0)
                    )
                else:
                    fallback_action = heuristic_resilience_action(env)
                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            (cfg.robust_uncertainty_threshold - uncertainty)
                            / cfg.robust_uncertainty_threshold,
                        ),
                    )
                    ppo_weight = 0.25 + 0.55 * confidence
                    ppo_candidate = float(
                        np.clip(
                            ppo_weight * ppo_action + (1.0 - ppo_weight) * fallback_action,
                            -1.0,
                            1.0,
                        )
                    )
                    if use_shield:
                        ppo_score = conservative_reward_estimate(env, ppo_candidate)
                        fallback_score = conservative_reward_estimate(env, fallback_action)
                        action = ppo_candidate if ppo_score >= fallback_score else fallback_action
                    else:
                        action = ppo_candidate

                action_t = torch.tensor([[action]], dtype=torch.float32)
                log_prob = dist.log_prob(action_t).sum(dim=-1).squeeze(0)
            else:
                action, log_prob = agent.select_action(encoded_np)
                if use_risk:
                    # Risk-only variant: keep pure PPO training objective but damp actions with target prior.
                    target_prior = float(np.clip(env.target_difficulty - env.difficulty, -1.0, 1.0))
                    action = float(np.clip(0.82 * action + 0.18 * target_prior, -1.0, 1.0))

            next_state, raw_reward, _, info = env.step(action)

            if use_risk:
                rolling_raw_rewards.append(float(raw_reward))
                tail = rolling_raw_rewards[-6:]
                variance_penalty = float(np.var(tail)) if len(tail) >= 2 else 0.0
                reward = float(raw_reward - cfg.robust_risk_lambda * variance_penalty)
            else:
                reward = float(raw_reward)

            states.append(encoded_np)
            actions.append(float(action))
            log_probs.append(log_prob)
            rewards.append(reward)
            raw_rewards.append(float(raw_reward))

            delays.append(info["delay"])
            attacks.append(info["attack_level"])
            target_errors.append(info["target_error"])
            fork_pressures.append(info["fork_pressure"])

            state = next_state

        returns = agent.compute_returns(rewards)
        agent.ppo_update(states, actions, log_probs, returns)

        episodes.append(
            {
                "reward": float(sum(raw_rewards)),
                "delay": float(statistics.mean(delays)),
                "attack": float(statistics.mean(attacks)),
                "target_error": float(statistics.mean(target_errors)),
                "fork_pressure": float(statistics.mean(fork_pressures)),
                "fallback_rate": float(fallback_steps / cfg.steps_per_episode),
                "uncertainty": float(statistics.mean(uncertainties)) if uncertainties else 0.0,
            }
        )

    return summarize_episode_stats(episodes)


def aggregate(per_seed: list[dict[str, float]]) -> dict[str, float]:
    keys = list(per_seed[0].keys())
    out: dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in per_seed]
        out[f"{key}_mean"] = float(statistics.mean(vals))
        out[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def main() -> int:
    cfg = Config()
    methods = [
        "random",
        "heuristic_target",
        "heuristic_resilience",
        "ppo_full",
        "robust_risk_only",
        "robust_fallback_only",
        "robust_ppo",
    ]

    results: list[dict[str, object]] = []
    for method in methods:
        per_seed: list[dict[str, float]] = []
        for seed in cfg.seeds:
            if method == "ppo_full":
                row = run_ppo(cfg, seed, mode="ppo_full")
            elif method == "robust_risk_only":
                row = run_ppo(cfg, seed, mode="robust_risk_only")
            elif method == "robust_fallback_only":
                row = run_ppo(cfg, seed, mode="robust_fallback_only")
            elif method == "robust_ppo":
                row = run_ppo(cfg, seed, robust=True, mode="robust_ppo")
            else:
                row = run_baseline(cfg, seed, method)
            per_seed.append(row)

        results.append(
            {
                "method": method,
                "per_seed": per_seed,
                "aggregate": aggregate(per_seed),
            }
        )

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "episodes": cfg.episodes,
            "steps_per_episode": cfg.steps_per_episode,
            "seeds": list(cfg.seeds),
            "start_difficulty": cfg.start_difficulty,
            "target_difficulty": cfg.target_difficulty,
            "attack_prob": cfg.attack_prob,
            "delay_noise": cfg.delay_noise,
            "burst_prob": cfg.burst_prob,
            "burst_scale": cfg.burst_scale,
            "robust_uncertainty_threshold": cfg.robust_uncertainty_threshold,
            "robust_risk_lambda": cfg.robust_risk_lambda,
            "robust_fallback_blend": cfg.robust_fallback_blend,
            "robust_curriculum_max_scale": cfg.robust_curriculum_max_scale,
        },
        "results": results,
    }

    out_json = ROOT / "experiments" / "results" / "adversarial_results.json"
    out_md = ROOT / "docs" / "ADVERSARIAL_RESULTS.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Adversarial Governance Results",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "| Method | Reward Last5 (mean±std) | Delay Last5 | Fork Pressure Last5 | Target Error Last5 | Fallback Rate Last5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        agg = row["aggregate"]
        lines.append(
            f"| {row['method']} | {agg['mean_reward_last5_mean']:.3f} ± {agg['mean_reward_last5_std']:.3f} "
            f"| {agg['mean_delay_last5_mean']:.3f} | {agg['mean_fork_pressure_last5_mean']:.3f} "
            f"| {agg['mean_target_error_last5_mean']:.3f} | {agg['mean_fallback_rate_last5_mean']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote adversarial results JSON: {out_json}")
    print(f"Wrote adversarial results markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())