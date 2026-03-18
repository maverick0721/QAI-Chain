import numpy as np


class BlockchainEnv:

    def __init__(self, blockchain):

        self.blockchain = blockchain

        self.state_dim = 5
        self.action_dim = 3  # actions map to {-1, 0, +1} difficulty delta
        self.min_difficulty = 1
        self.max_difficulty = 8
        self.target_difficulty = 3

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
        # Convert discrete action index to symmetric difficulty change.
        action_idx = int(action[0])
        action_map = {0: -1, 1: 0, 2: 1}
        delta = action_map.get(action_idx, 0)

        prev_difficulty = int(self.blockchain.difficulty)
        next_difficulty = int(np.clip(
            prev_difficulty + delta,
            self.min_difficulty,
            self.max_difficulty,
        ))
        self.blockchain.difficulty = next_difficulty

        reward = self.compute_reward(delta)

        next_state = self.get_state()

        done = False

        return next_state, reward, done

    def compute_reward(self, delta):

        chain = self.blockchain.chain

        num_blocks = len(chain)

        block_sizes = [len(b.transactions) for b in chain]

        throughput = sum(block_sizes) / max(1, num_blocks)

        stability = 1.0  # placeholder

        # Keep reward scale smooth and make a clear optimum around target difficulty.
        throughput_bonus = 0.1 * min(throughput, 10.0)
        difficulty_penalty = 0.2 * abs(self.blockchain.difficulty - self.target_difficulty)
        action_penalty = 0.05 * abs(delta)

        reward = stability + throughput_bonus - difficulty_penalty - action_penalty

        return reward