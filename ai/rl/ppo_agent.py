import torch
import torch.nn as nn
import torch.optim as optim


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

    def select_action(self, state):

        state = torch.tensor(state).float().unsqueeze(0)

        logits = self.policy(state)

        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards):

        returns = []

        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns)