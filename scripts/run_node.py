from pathlib import Path
import sys

# Allow running this script directly from scripts/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.blockchain.blockchain import Blockchain
from core.blockchain.miner import proof_of_work
from core.blockchain.transaction import Transaction
from core.blockchain.block import Block


blockchain = Blockchain()

tx = Transaction("alice", "bob", 5)

block = Block(
    index=len(blockchain.chain),
    transactions=[tx],
    previous_hash=blockchain.last_block().hash
)

block = proof_of_work(block, blockchain.difficulty)

blockchain.add_block(block)

print("Block mined:", block.hash)
