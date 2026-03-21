import torch

from ai.data.blockchain_dataset import BlockchainDataset
from ai.models.metrics_encoder import MetricsEncoder
from ai.models.policy_network import PolicyNetwork


def run_ai(blockchain):

    dataset = BlockchainDataset(blockchain)
    features = dataset.extract_features()

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    encoder = MetricsEncoder()
    policy = PolicyNetwork()

    encoded = encoder(x)
    action_logits = policy(encoded)

    return action_logits