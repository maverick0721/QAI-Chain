from core.blockchain.block import Block
from core.utils.logger import get_logger

logger = get_logger("blockchain")


class Blockchain:

    def __init__(self, difficulty=3):

        self.chain = []
        self.difficulty = difficulty

        self.create_genesis_block()

    def create_genesis_block(self):

        genesis = Block(0, [], "0")

        self.chain.append(genesis)

    def last_block(self):

        return self.chain[-1]

    def add_block(self, block):

        self.chain.append(block)

    def is_valid_block(self, block, previous_block):
        if previous_block.hash != block.previous_hash:
            return False

        if not block.hash.startswith("0" * self.difficulty):
            return False

        if block.compute_hash() != block.hash:
            return False

        return True