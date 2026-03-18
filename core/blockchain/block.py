from core.utils import sha256, serialize, current_timestamp


class Block:

    def __init__(self, index, transactions, previous_hash, zk_proof=None):

        self.index = index
        self.timestamp = current_timestamp()
        self.transactions = transactions
        self.previous_hash = previous_hash

        self.nonce = 0
        self.zk_proof = zk_proof  # NEW FIELD

        self.hash = self.compute_hash()

    def compute_hash(self):

        block_data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [t.to_dict() for t in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "zk_proof": self.zk_proof  # INCLUDED IN HASH
        }

        return sha256(serialize(block_data))