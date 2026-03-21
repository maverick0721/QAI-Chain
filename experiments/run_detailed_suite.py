from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ai.models.metrics_encoder import MetricsEncoder
from ai.models.policy_network import PolicyNetwork
from ai.models.value_network import ValueNetwork
from ai.rl.environment import BlockchainEnv
from ai.rl.ppo_agent import PPOAgent
from core.blockchain.blockchain import Blockchain


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    episodes: int = 60
    steps_per_episode: int = 25
    seeds: tuple[int, ...] = (3, 7, 11, 17, 21, 42, 84, 126, 5, 9, 13, 19, 23, 31, 57, 99)
    start_difficulty: int = 7
    target_difficulty: int = 3
    entropy_start: float = 0.01
    entropy_end: float = 0.002


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_baseline(cfg: Config, seed: int, method: str) -> dict[str, object]:
    set_seed(seed)
    blockchain = Blockchain(difficulty=cfg.start_difficulty)
    env = BlockchainEnv(blockchain)

    rewards: list[float] = []
    difficulties: list[int] = []

    for _ in range(cfg.episodes):
        _ = env.reset()
        total_reward = 0.0

        for _ in range(cfg.steps_per_episode):
            if method == "random":
                action = float(np.random.uniform(-1.0, 1.0))
            elif method == "heuristic_target":
                action = float(np.clip(env.target_difficulty - env.blockchain.difficulty, -1.0, 1.0))
            else:
                raise ValueError(method)

            _, reward, _ = env.step(action)
            total_reward += float(reward)

        rewards.append(total_reward)
        difficulties.append(int(blockchain.difficulty))

    return {
        "episode_rewards": rewards,
        "episode_difficulties": difficulties,
    }


def run_ppo(cfg: Config, seed: int, normalize_adv: bool = True) -> dict[str, object]:
    set_seed(seed)
    blockchain = Blockchain(difficulty=cfg.start_difficulty)
    env = BlockchainEnv(blockchain)

    encoder = MetricsEncoder()
    policy = PolicyNetwork()
    value = ValueNetwork()
    agent = PPOAgent(policy, value)

    rewards_by_episode: list[float] = []
    difficulties_by_episode: list[int] = []

    for ep in range(cfg.episodes):
        progress = ep / max(1, cfg.episodes - 1)
        agent.entropy_coef = cfg.entropy_start + (cfg.entropy_end - cfg.entropy_start) * progress

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

        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32))
        actions_t = torch.as_tensor(actions, dtype=torch.float32).unsqueeze(-1)
        old_log_probs_t = torch.stack(log_probs).detach().view(-1)
        returns_t = returns.detach().float().view(-1)

        value_pred = agent.value(states_t).squeeze(-1)
        advantages = returns_t - value_pred.detach()
        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean, std = agent.policy(states_t)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs_t)
        clipped_ratio = torch.clamp(ratio, 1 - agent.eps_clip, 1 + agent.eps_clip)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = (returns_t - value_pred).pow(2).mean()
        loss = policy_loss + agent.value_coef * value_loss - agent.entropy_coef * entropy

        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(agent.policy.parameters()) + list(agent.value.parameters()),
            agent.max_grad_norm,
        )
        agent.optimizer.step()

        rewards_by_episode.append(float(sum(rewards)))
        difficulties_by_episode.append(int(blockchain.difficulty))

    return {
        "episode_rewards": rewards_by_episode,
        "episode_difficulties": difficulties_by_episode,
    }


def main() -> int:
    cfg = Config()
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "episodes": cfg.episodes,
            "steps_per_episode": cfg.steps_per_episode,
            "seeds": list(cfg.seeds),
            "start_difficulty": cfg.start_difficulty,
            "target_difficulty": cfg.target_difficulty,
        },
        "methods": {
            "random": {},
            "heuristic_target": {},
            "ppo_full": {},
            "ppo_no_adv_norm": {},
        },
    }

    for seed in cfg.seeds:
        out["methods"]["random"][str(seed)] = run_baseline(cfg, seed, "random")
        out["methods"]["heuristic_target"][str(seed)] = run_baseline(cfg, seed, "heuristic_target")
        out["methods"]["ppo_full"][str(seed)] = run_ppo(cfg, seed, normalize_adv=True)
        out["methods"]["ppo_no_adv_norm"][str(seed)] = run_ppo(cfg, seed, normalize_adv=False)

    out_path = ROOT / "experiments" / "results" / "detailed_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote detailed results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())