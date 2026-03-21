import torch

from ai.data.blockchain_dataset import BlockchainDataset
from ai.pipeline import run_ai
from core.blockchain.blockchain import Blockchain


def test_run_ai_returns_policy_outputs_with_expected_shapes():

    torch.manual_seed(42)
    blockchain = Blockchain()

    mean, std = run_ai(blockchain)

    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert mean.shape == (1, 1)
    assert std.shape == (1,)
    assert torch.isfinite(mean).all().item()
    assert torch.isfinite(std).all().item()


def test_dataset_genesis_only_returns_zero_feature_vector():

    blockchain = Blockchain()
    features = BlockchainDataset(blockchain).extract_features()

    assert features.shape == (5,)
    assert float(features.sum()) == 0.0