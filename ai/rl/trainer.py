import torch

from ai.rl.environment import BlockchainEnv
from ai.models.metrics_encoder import MetricsEncoder
from ai.models.policy_network import PolicyNetwork
from ai.models.value_network import ValueNetwork
from ai.rl.ppo_agent import PPOAgent


def train(blockchain, episodes=50):

    env = BlockchainEnv(blockchain)

    encoder = MetricsEncoder()
    policy = PolicyNetwork()
    value = ValueNetwork()

    agent = PPOAgent(policy, value)

    for ep in range(episodes):

        state = env.reset()

        rewards = []
        log_probs = []

        for step in range(20):

            encoded = encoder(torch.tensor(state).float().unsqueeze(0))

            action, log_prob = agent.select_action(encoded.detach().numpy()[0])

            next_state, reward, done = env.step([action])

            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        returns = agent.compute_returns(rewards)

        loss = -returns.mean()

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        print(f"Episode {ep} | Reward: {sum(rewards):.2f}")