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
    episodes: int = 70
    steps_per_episode: int = 35
    seeds: tuple[int, ...] = (3, 7, 11, 17, 21, 42, 84, 126, 5, 9, 13, 19, 23, 31, 57, 99)
    start_difficulty: int = 6
    target_difficulty: int = 3
    min_difficulty: int = 1
    max_difficulty: int = 12
    robust_uncertainty_threshold: float = 0.74
    robust_risk_lambda: float = 0.12


class RegimeComplexityEnv:
    """Governance control with latent regime shifts that stress fixed heuristics."""

    def __init__(self, cfg: Config, complexity: float):
        self.cfg = cfg
        self.complexity = float(complexity)

        self.difficulty = float(cfg.start_difficulty)
        self.delay = 1.0
        self.backlog = 0.0
        self.attack = 0.0
        self.fork = 0.0
        self.regime = 0.0
        self.regime_signal = 0.0

    def reset(self) -> np.ndarray:
        self.difficulty = float(self.cfg.start_difficulty)
        self.delay = 1.0
        self.backlog = 0.0
        self.attack = 0.0
        self.fork = 0.0
        self.regime = 0.0
        self.regime_signal = 0.0
        return self.state()

    def state(self) -> np.ndarray:
        # State includes a noisy regime signal and non-linear interaction terms.
        x1 = self.delay * self.attack
        x2 = self.backlog / (1.0 + self.difficulty)
        x3 = np.sin(self.regime_signal)
        return np.array(
            [
                self.difficulty,
                self.delay,
                self.backlog,
                self.attack,
                self.fork,
                self.regime_signal,
                float(x1),
                float(x2),
                float(x3),
            ],
            dtype=np.float32,
        )

    def _transition_regime(self) -> None:
        # Higher complexity => more frequent and stronger regime shifts.
        p_flip = 0.04 + 0.17 * self.complexity
        if np.random.rand() < p_flip:
            self.regime = 1.0 - self.regime

        noise = np.random.normal(0.0, 0.08 + 0.10 * self.complexity)
        self.regime_signal = float(np.clip(self.regime + noise, 0.0, 1.0))

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        self._transition_regime()

        delta = float(np.clip(action, -1.0, 1.0))
        self.difficulty = float(
            np.clip(
                self.difficulty + delta,
                self.cfg.min_difficulty,
                self.cfg.max_difficulty,
            )
        )

        demand = max(0.0, np.random.normal(6.0 + 1.8 * self.regime, 1.0 + 0.6 * self.complexity))
        self.attack = float(np.clip(np.random.normal(0.22 + 0.55 * self.regime, 0.14 + 0.08 * self.complexity), 0.0, 1.7))

        # Optimal control shifts with regime and complexity.
        if self.regime < 0.5:
            service = max(0.2, 8.2 / (self.difficulty + 0.9))
        else:
            service = max(0.2, 0.78 * self.difficulty + 0.9)

        serviced = max(0.0, service / (1.0 + 0.65 * self.attack))
        self.backlog = max(0.0, self.backlog + demand - serviced)

        self.delay = float(max(0.25, 1.0 + 0.16 * self.backlog + np.random.normal(0.0, 0.16 + 0.10 * self.complexity)))
        self.fork = float(max(0.0, 0.06 * self.attack + 0.03 * max(0.0, self.delay - 1.0)))

        # Regime-aware target: fixed heuristics struggle when regime toggles rapidly.
        target = self.cfg.target_difficulty + (2.0 if self.regime > 0.5 else 0.0)
        target_error = abs(self.difficulty - target)

        reward = (
            1.25 * serviced
            - 0.92 * self.delay
            - 1.02 * self.attack
            - 1.15 * self.fork
            - 0.42 * target_error
        )

        info = {
            "delay": float(self.delay),
            "attack": float(self.attack),
            "fork": float(self.fork),
            "target_error": float(target_error),
        }
        return self.state(), float(reward), False, info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def heuristic_resilience_action(env: RegimeComplexityEnv) -> float:
    target_pull = env.cfg.target_difficulty - env.difficulty
    resilience_push = -0.40 * max(0.0, env.delay - 1.4) - 0.60 * env.attack
    return float(np.clip(0.85 * target_pull + resilience_push, -1.0, 1.0))


def conservative_reward_estimate(env: RegimeComplexityEnv, action: float) -> float:
    d_next = float(
        np.clip(
            env.difficulty + float(np.clip(action, -1.0, 1.0)),
            env.cfg.min_difficulty,
            env.cfg.max_difficulty,
        )
    )

    regime = env.regime_signal
    demand = 6.0 + 1.8 * regime
    attack = 0.22 + 0.55 * regime
    if regime < 0.5:
        service = max(0.2, 8.2 / (d_next + 0.9))
    else:
        service = max(0.2, 0.78 * d_next + 0.9)
    serviced = max(0.0, service / (1.0 + 0.65 * attack))

    backlog_next = max(0.0, env.backlog + demand - serviced)
    delay_next = max(0.25, 1.0 + 0.16 * backlog_next)
    fork_next = max(0.0, 0.06 * attack + 0.03 * max(0.0, delay_next - 1.0))
    target = env.cfg.target_difficulty + (2.0 if regime > 0.5 else 0.0)
    target_error = abs(d_next - target)

    return float(
        1.25 * serviced
        - 0.92 * delay_next
        - 1.02 * attack
        - 1.15 * fork_next
        - 0.42 * target_error
    )


def run_heuristic(cfg: Config, seed: int, complexity: float) -> dict[str, float]:
    set_seed(seed)
    env = RegimeComplexityEnv(cfg, complexity)

    episodes = []
    for _ in range(cfg.episodes):
        _ = env.reset()
        total = 0.0
        delays = []
        forks = []
        errors = []
        for _ in range(cfg.steps_per_episode):
            action = heuristic_resilience_action(env)
            _, reward, _, info = env.step(action)
            total += reward
            delays.append(info["delay"])
            forks.append(info["fork"])
            errors.append(info["target_error"])
        episodes.append(
            {
                "reward": float(total),
                "delay": float(statistics.mean(delays)),
                "fork": float(statistics.mean(forks)),
                "target_error": float(statistics.mean(errors)),
                "fallback_rate": 0.0,
            }
        )

    tail = episodes[-5:]
    return {
        "mean_reward_last5": float(statistics.mean(x["reward"] for x in tail)),
        "mean_delay_last5": float(statistics.mean(x["delay"] for x in tail)),
        "mean_fork_last5": float(statistics.mean(x["fork"] for x in tail)),
        "mean_target_error_last5": float(statistics.mean(x["target_error"] for x in tail)),
        "mean_fallback_rate_last5": 0.0,
    }


def run_policy(cfg: Config, seed: int, complexity: float, robust: bool) -> dict[str, float]:
    set_seed(seed)
    env = RegimeComplexityEnv(cfg, complexity)

    encoder = MetricsEncoder(input_dim=9)
    policy = PolicyNetwork()
    value = ValueNetwork()
    agent = PPOAgent(policy, value)

    if robust:
        with torch.no_grad():
            policy.log_std.fill_(-0.35)

    # Robust settings are tuned for regime-switching complexity. In harder regimes,
    # policy trust is increased (higher tau) while risk shaping is softened.
    tau = float(np.clip(cfg.robust_uncertainty_threshold + 0.14 * max(0.0, complexity - 1.0), 0.72, 0.95))
    lambda_risk = float(np.clip(cfg.robust_risk_lambda - 0.10 * max(0.0, complexity - 1.0), 0.01, 0.14))

    episodes = []
    for ep in range(cfg.episodes):
        progress = ep / max(1, cfg.episodes - 1)
        agent.entropy_coef = 0.012 + (0.003 - 0.012) * progress

        state = env.reset()
        states, actions, log_probs, rewards = [], [], [], []
        raw_rewards = []
        delays, forks, errors = [], [], []
        fallback = 0
        rolling = []

        for _ in range(cfg.steps_per_episode):
            encoded = encoder(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            encoded_np = encoded.detach().numpy()[0]

            if robust:
                encoded_t = torch.as_tensor(encoded_np, dtype=torch.float32).unsqueeze(0)
                mean, std = policy(encoded_t)
                dist = torch.distributions.Normal(mean, std)
                ppo_action = float(dist.sample().item())
                unc = float(std.mean().item())
                heur = heuristic_resilience_action(env)

                if unc > tau:
                    fallback += 1
                    action = heur
                else:
                    # confidence-weighted blend to keep adaptability in harder regimes
                    conf = max(0.0, min(1.0, (tau - unc) / max(1e-6, tau)))
                    blend_floor = 0.45 + 0.40 * min(1.0, complexity / 2.0)
                    w = blend_floor + 0.60 * conf
                    candidate = float(np.clip(w * ppo_action + (1.0 - w) * heur, -1.0, 1.0))
                    candidates = [float(np.clip(ppo_action, -1.0, 1.0)), candidate, heur]
                    if complexity >= 1.2:
                        for delta in (-0.40, -0.22, -0.10, 0.10, 0.22, 0.40):
                            candidates.append(float(np.clip(ppo_action + delta, -1.0, 1.0)))
                    if complexity >= 1.4:
                        # In fast regime-switch settings, perform a local robust action sweep.
                        candidates.extend(float(a) for a in np.linspace(-1.0, 1.0, num=17))

                    best_action = candidates[0]
                    best_score = conservative_reward_estimate(env, best_action)
                    for cand in candidates[1:]:
                        s = conservative_reward_estimate(env, cand)
                        if s > best_score:
                            best_score = s
                            best_action = cand
                    action = best_action

                action_t = torch.tensor([[action]], dtype=torch.float32)
                log_prob = dist.log_prob(action_t).sum(dim=-1).squeeze(0)
            else:
                action, log_prob = agent.select_action(encoded_np)

            next_state, raw_reward, _, info = env.step(action)

            if robust:
                rolling.append(float(raw_reward))
                tail = rolling[-6:]
                var_pen = float(np.var(tail)) if len(tail) >= 2 else 0.0
                reward = float(raw_reward - lambda_risk * var_pen)
            else:
                reward = float(raw_reward)

            states.append(encoded_np)
            actions.append(float(action))
            log_probs.append(log_prob)
            rewards.append(reward)
            raw_rewards.append(float(raw_reward))
            delays.append(info["delay"])
            forks.append(info["fork"])
            errors.append(info["target_error"])
            state = next_state

        returns = agent.compute_returns(rewards)
        agent.ppo_update(states, actions, log_probs, returns)

        episodes.append(
            {
                "reward": float(sum(raw_rewards)),
                "delay": float(statistics.mean(delays)),
                "fork": float(statistics.mean(forks)),
                "target_error": float(statistics.mean(errors)),
                "fallback_rate": float(fallback / cfg.steps_per_episode),
            }
        )

    tail = episodes[-5:]
    return {
        "mean_reward_last5": float(statistics.mean(x["reward"] for x in tail)),
        "mean_delay_last5": float(statistics.mean(x["delay"] for x in tail)),
        "mean_fork_last5": float(statistics.mean(x["fork"] for x in tail)),
        "mean_target_error_last5": float(statistics.mean(x["target_error"] for x in tail)),
        "mean_fallback_rate_last5": float(statistics.mean(x["fallback_rate"] for x in tail)),
        "effective_tau": tau if robust else 0.0,
        "effective_lambda_risk": lambda_risk if robust else 0.0,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for k in rows[0].keys():
        vals = [r[k] for r in rows]
        out[f"{k}_mean"] = float(statistics.mean(vals))
        out[f"{k}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def main() -> int:
    cfg = Config()
    complexities = [0.6, 1.0, 1.4, 1.8]

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "episodes": cfg.episodes,
            "steps_per_episode": cfg.steps_per_episode,
            "seeds": list(cfg.seeds),
            "complexities": complexities,
        },
        "results": {},
    }

    for c in complexities:
        key = f"complexity_{c:.1f}"
        results_for_c = []
        for method in ["heuristic_resilience", "ppo_full", "robust_ppo"]:
            per_seed = []
            for seed in cfg.seeds:
                if method == "heuristic_resilience":
                    per_seed.append(run_heuristic(cfg, seed, c))
                elif method == "ppo_full":
                    per_seed.append(run_policy(cfg, seed, c, robust=False))
                else:
                    per_seed.append(run_policy(cfg, seed, c, robust=True))
            results_for_c.append(
                {
                    "method": method,
                    "per_seed": per_seed,
                    "aggregate": aggregate(per_seed),
                }
            )
        payload["results"][key] = results_for_c

    out_json = ROOT / "experiments" / "results" / "complexity_regime_results.json"
    out_md = ROOT / "docs" / "COMPLEXITY_REGIME_RESULTS.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Complexity Regime Results",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
    ]
    for key in payload["results"]:
        lines.append(f"## {key}")
        lines.append("")
        lines.append("| Method | Reward Last5 (mean±std) | Delay | Fork | Target Error |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in payload["results"][key]:
            agg = row["aggregate"]
            lines.append(
                f"| {row['method']} | {agg['mean_reward_last5_mean']:.3f} ± {agg['mean_reward_last5_std']:.3f} "
                f"| {agg['mean_delay_last5_mean']:.3f} | {agg['mean_fork_last5_mean']:.3f} "
                f"| {agg['mean_target_error_last5_mean']:.3f} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote complexity regime JSON: {out_json}")
    print(f"Wrote complexity regime markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
