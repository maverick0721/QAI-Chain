from core.blockchain.block import Block
from core.utils.logger import get_logger
from crypto.pqc.verification import verify_transaction

logger = get_logger("blockchain")


class Blockchain:

    def __init__(self, difficulty=3):

        self.chain = []
        self.difficulty = difficulty
        self.audit_trail: list[dict[str, object]] = []

        self.create_genesis_block()

    
    # Genesis Block
    
    def create_genesis_block(self):

        genesis = Block(0, [], "0")

        self.chain.append(genesis)

    
    # Get last block
    
    def last_block(self):

        return self.chain[-1]

    
    # Add block
    
    def add_block(self, block):

        self.chain.append(block)

    
    # Transaction Validation (PQC)
    
    def validate_transaction(self, tx):

        # sender is treated as public key
        return verify_transaction(tx, tx.sender)

    
    # Block Validation
    
    def is_valid_block(self, block, previous_block):

        # check previous hash link
        if previous_block.hash != block.previous_hash:
            return False

        # check PoW
        if not block.hash.startswith("0" * self.difficulty):
            return False

        # check hash integrity
        if block.compute_hash() != block.hash:
            return False

        # check transactions
        for tx in block.transactions:
            if not self.validate_transaction(tx):
                return False

        return True

    def commit_audit_record(self, record: dict[str, object]) -> str:
        payload = dict(record)
        payload["audit_index"] = len(self.audit_trail)
        self.audit_trail.append(payload)
        return str(payload.get("record_hash", payload["audit_index"]))