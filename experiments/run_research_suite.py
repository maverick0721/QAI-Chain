from __future__ import annotations

import json
import argparse
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.models.metrics_encoder import MetricsEncoder
from ai.models.policy_network import PolicyNetwork
from ai.models.value_network import ValueNetwork
from ai.rl.environment import BlockchainEnv
from ai.rl.ppo_agent import PPOAgent
from core.blockchain.blockchain import Blockchain


@dataclass
class SuiteConfig:
    episodes: int = 20
    steps_per_episode: int = 20
    seeds: tuple[int, ...] = (11, 21, 42)
    start_difficulty: int = 7
    target_difficulty: int = 3
    ppo_entropy_start: float = 0.01
    ppo_entropy_end: float = 0.002


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _episode_stats(
    rewards: list[float],
    final_difficulty: int,
    target_difficulty: int,
) -> dict[str, float]:
    tail = rewards[-5:] if len(rewards) >= 5 else rewards
    mean_last5 = float(statistics.mean(tail)) if tail else 0.0
    std_last5 = float(statistics.pstdev(tail)) if len(tail) > 1 else 0.0

    return {
        "mean_reward_last5": mean_last5,
        "mean_reward_all": float(statistics.mean(rewards)) if rewards else 0.0,
        "best_reward": float(max(rewards)) if rewards else 0.0,
        "reward_std_last5": std_last5,
        "final_difficulty": float(final_difficulty),
        "abs_target_error": float(abs(final_difficulty - target_difficulty)),
    }


def _aggregate(metrics_per_seed: list[dict[str, float]]) -> dict[str, float]:
    keys = list(metrics_per_seed[0].keys())
    out: dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in metrics_per_seed]
        out[f"{key}_mean"] = float(statistics.mean(vals))
        out[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def _run_baseline(
    cfg: SuiteConfig,
    seed: int,
    action_fn: Callable[[BlockchainEnv], float],
) -> dict[str, float]:
    set_seed(seed)
    blockchain = Blockchain(difficulty=cfg.start_difficulty)
    env = BlockchainEnv(blockchain)

    episode_rewards: list[float] = []
    for _ in range(cfg.episodes):
        _ = env.reset()
        total_reward = 0.0

        for _ in range(cfg.steps_per_episode):
            action = action_fn(env)
            _, reward, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    return _episode_stats(episode_rewards, blockchain.difficulty, cfg.target_difficulty)


def _ppo_update_custom(
    agent: PPOAgent,
    states: list[np.ndarray],
    actions: list[float],
    old_log_probs: list[torch.Tensor],
    returns: torch.Tensor,
    normalize_advantage: bool,
) -> None:
    states_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32))
    actions_tensor = torch.as_tensor(actions, dtype=torch.float32).unsqueeze(-1)
    old_log_probs_tensor = torch.stack(old_log_probs).detach().view(-1)
    returns_tensor = returns.detach().float().view(-1)

    value_pred = agent.value(states_tensor).squeeze(-1)
    advantages = returns_tensor - value_pred.detach()
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    mean, std = agent.policy(states_tensor)
    dist = torch.distributions.Normal(mean, std)

    new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1).mean()

    ratio = torch.exp(new_log_probs - old_log_probs_tensor)
    clipped_ratio = torch.clamp(ratio, 1 - agent.eps_clip, 1 + agent.eps_clip)

    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = (returns_tensor - value_pred).pow(2).mean()
    loss = policy_loss + agent.value_coef * value_loss - agent.entropy_coef * entropy

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.policy.parameters()) + list(agent.value.parameters()),
        agent.max_grad_norm,
    )
    agent.optimizer.step()


def _run_ppo(
    cfg: SuiteConfig,
    seed: int,
    entropy_anneal: bool,
    normalize_advantage: bool,
) -> dict[str, float]:
    set_seed(seed)
    blockchain = Blockchain(difficulty=cfg.start_difficulty)
    env = BlockchainEnv(blockchain)

    encoder = MetricsEncoder()
    policy = PolicyNetwork()
    value = ValueNetwork()
    agent = PPOAgent(policy, value)

    episode_rewards: list[float] = []

    for ep in range(cfg.episodes):
        if entropy_anneal:
            progress = ep / max(1, cfg.episodes - 1)
            agent.entropy_coef = (
                cfg.ppo_entropy_start
                + (cfg.ppo_entropy_end - cfg.ppo_entropy_start) * progress
            )
        else:
            agent.entropy_coef = cfg.ppo_entropy_start

        state = env.reset()

        states: list[np.ndarray] = []
        actions: list[float] = []
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []

        for _ in range(cfg.steps_per_episode):
            encoded = encoder(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action, log_prob = agent.select_action(encoded.detach().numpy()[0])

            next_state, reward, _ = env.step(action)

            states.append(encoded.detach().numpy()[0])
            actions.append(float(action))
            log_probs.append(log_prob)
            rewards.append(float(reward))

            state = next_state

        returns = agent.compute_returns(rewards)
        _ppo_update_custom(
            agent,
            states,
            actions,
            log_probs,
            returns,
            normalize_advantage=normalize_advantage,
        )

        episode_rewards.append(float(sum(rewards)))

    return _episode_stats(episode_rewards, blockchain.difficulty, cfg.target_difficulty)


def _run_method(cfg: SuiteConfig, method: str) -> dict[str, object]:
    per_seed: list[dict[str, float]] = []

    for seed in cfg.seeds:
        if method == "random":
            result = _run_baseline(cfg, seed, lambda _env: float(np.random.uniform(-1.0, 1.0)))
        elif method == "heuristic_target":
            result = _run_baseline(
                cfg,
                seed,
                lambda env: float(
                    np.clip(
                        env.target_difficulty - env.blockchain.difficulty,
                        -1.0,
                        1.0,
                    )
                ),
            )
        elif method == "ppo_full":
            result = _run_ppo(cfg, seed, entropy_anneal=True, normalize_advantage=True)
        elif method == "ppo_no_entropy_anneal":
            result = _run_ppo(cfg, seed, entropy_anneal=False, normalize_advantage=True)
        elif method == "ppo_no_adv_norm":
            result = _run_ppo(cfg, seed, entropy_anneal=True, normalize_advantage=False)
        else:
            raise ValueError(f"Unknown method: {method}")

        per_seed.append(result)

    return {
        "method": method,
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }


def _render_markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    lines = [
        "# Research Results",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "## Configuration",
        "",
        f"- episodes: {payload['config']['episodes']}",
        f"- steps_per_episode: {payload['config']['steps_per_episode']}",
        f"- seeds: {payload['config']['seeds']}",
        f"- start_difficulty: {payload['config']['start_difficulty']}",
        "",
        "## Main Comparison",
        "",
        "| Method | Mean Reward (Last5) | Std Across Seeds | Final Abs Target Error |",
        "|---|---:|---:|---:|",
    ]

    for row in results:
        method = row["method"]
        agg = row["aggregate"]
        if method not in {"random", "heuristic_target", "ppo_full"}:
            continue
        lines.append(
            "| "
            f"{method} | {agg['mean_reward_last5_mean']:.3f} | "
            f"{agg['mean_reward_last5_std']:.3f} | {agg['abs_target_error_mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## PPO Ablation",
            "",
            "| Variant | Mean Reward (Last5) | Std Across Seeds |",
            "|---|---:|---:|",
        ]
    )

    for row in results:
        method = row["method"]
        if not method.startswith("ppo"):
            continue
        agg = row["aggregate"]
        lines.append(
            "| "
            f"{method} | {agg['mean_reward_last5_mean']:.3f} | {agg['mean_reward_last5_std']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QAI-Chain research suite")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="11,21,42")
    parser.add_argument("--start-difficulty", type=int, default=7)
    parser.add_argument("--target-difficulty", type=int, default=3)
    parser.add_argument("--out-json", type=str, default="experiments/results/research_results.json")
    parser.add_argument("--out-md", type=str, default="docs/RESEARCH_RESULTS.md")
    args = parser.parse_args()

    parsed_seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    cfg = SuiteConfig(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seeds=parsed_seeds,
        start_difficulty=args.start_difficulty,
        target_difficulty=args.target_difficulty,
    )

    methods = [
        "random",
        "heuristic_target",
        "ppo_full",
        "ppo_no_entropy_anneal",
        "ppo_no_adv_norm",
    ]

    result_rows = [_run_method(cfg, method) for method in methods]

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "episodes": cfg.episodes,
            "steps_per_episode": cfg.steps_per_episode,
            "seeds": list(cfg.seeds),
            "start_difficulty": cfg.start_difficulty,
            "target_difficulty": cfg.target_difficulty,
            "ppo_entropy_start": cfg.ppo_entropy_start,
            "ppo_entropy_end": cfg.ppo_entropy_end,
        },
        "results": result_rows,
    }

    out_json_path = ROOT / args.out_json
    out_md_path = ROOT / args.out_md

    out_dir = out_json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"Wrote research results JSON: {out_json_path}")
    print(f"Wrote research results markdown: {out_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())