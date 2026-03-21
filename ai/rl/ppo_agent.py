import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPOAgent:

    def __init__(self, policy, value, lr=3e-4):

        self.policy = policy
        self.value = value

        self.optimizer = optim.Adam(
            list(policy.parameters()) + list(value.parameters()),
            lr=lr
        )

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def select_action(self, state):

        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        mean, std = self.policy(state)

        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.item(), log_prob

    def compute_returns(self, rewards):

        returns: list[float] = []

        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns)

    def ppo_update(self, states, actions, old_log_probs, returns):

        states = torch.from_numpy(np.asarray(states, dtype=np.float32))
        actions = torch.as_tensor(actions, dtype=torch.float32).unsqueeze(-1)
        old_log_probs = torch.stack(old_log_probs).detach().view(-1)
        returns = returns.detach().float().view(-1)

        value_pred = self.value(states).squeeze(-1)
        advantages = returns - value_pred.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean, std = self.policy(states)

        dist = torch.distributions.Normal(mean, std)

        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        value_loss = (returns - value_pred).pow(2).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()
            