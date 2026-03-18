import numpy as np


class BlockchainEnv:

    def __init__(self, blockchain):

        self.blockchain = blockchain

        self.state_dim = 5
        self.action_dim = 3  # e.g., adjust difficulty

    def reset(self):

        return self.get_state()

    def get_state(self):

        chain = self.blockchain.chain

        num_blocks = len(chain)

        if num_blocks < 2:
            return np.zeros(self.state_dim)

        block_sizes = [len(b.transactions) for b in chain]

        avg_block_size = np.mean(block_sizes)
        tx_count = sum(block_sizes)

        avg_tx_per_block = tx_count / num_blocks

        fork_rate = 0.0  # placeholder
        latency = 1.0    # placeholder

        return np.array([
            num_blocks,
            avg_block_size,
            avg_tx_per_block,
            fork_rate,
            latency
        ], dtype=np.float32)

    def step(self, action):

       
        # Apply action
        # action[0] → difficulty adjustment
        delta = action[0]

        self.blockchain.difficulty = max(1, int(self.blockchain.difficulty + delta))

       
        # Simulate reward
       
        reward = self.compute_reward()

        next_state = self.get_state()

        done = False

        return next_state, reward, done

    def compute_reward(self):

        chain = self.blockchain.chain

        num_blocks = len(chain)

        block_sizes = [len(b.transactions) for b in chain]

        throughput = sum(block_sizes) / max(1, num_blocks)

        stability = 1.0  # placeholder

        # RESEARCH IDEA: multi-objective reward
        reward = throughput - 0.1 * self.blockchain.difficulty + stability

        return reward