import requests
from core.logger import get_logger

logger = get_logger("p2p")


class P2PNode:

    def __init__(self, address, peer_manager):
        self.address = address
        self.peer_manager = peer_manager

    def broadcast_transaction(self, tx):

        for peer in self.peer_manager.get_peers():
            try:
                requests.post(f"{peer}/transaction", json=tx.to_dict())
            except Exception as e:
                logger.warning(f"Failed to send tx to {peer}: {e}")

    def broadcast_block(self, block):

        for peer in self.peer_manager.get_peers():
            try:
                requests.post(f"{peer}/block", json={
                    "index": block.index,
                    "transactions": [t.to_dict() for t in block.transactions],
                    "previous_hash": block.previous_hash,
                    "nonce": block.nonce,
                    "hash": block.hash
                })
            except Exception as e:
                logger.warning(f"Failed to send block to {peer}: {e}")