import numpy as np


class BlockchainDataset:

    def __init__(self, blockchain):
        self.blockchain = blockchain

    def extract_features(self):

        chain = self.blockchain.chain

        num_blocks = len(chain)

        if num_blocks < 2:
            return np.zeros(5, dtype=np.float32)

        block_sizes = [len(b.transactions) for b in chain]
        avg_block_size = np.mean(block_sizes)

        tx_counts = sum(block_sizes)

        # simple proxy metrics
        avg_tx_per_block = tx_counts / num_blocks

        # pseudo metrics (we'll improve later)
        fork_rate = 0.0  # placeholder
        latency = 1.0    # placeholder

        return np.array([
            num_blocks,
            avg_block_size,
            avg_tx_per_block,
            fork_rate,
            latency
        ], dtype=np.float32)