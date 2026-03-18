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

        states = []
        actions = []
        log_probs = []
        rewards = []

        for step in range(30):

            encoded = encoder(torch.tensor(state).float().unsqueeze(0))

            action, log_prob = agent.select_action(encoded.detach().numpy()[0])

            next_state, reward, done = env.step(action)

            states.append(encoded.detach().numpy()[0])
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        returns = agent.compute_returns(rewards)

        agent.ppo_update(states, actions, log_probs, returns)

        print(f"Episode {ep} | Reward: {sum(rewards):.2f}")