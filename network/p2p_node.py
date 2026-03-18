import requests
from core.utils.logger import get_logger

logger = get_logger("p2p")


class P2PNode:

    def __init__(self, address, peer_manager):
        self.address = address
        self.peer_manager = peer_manager

    def broadcast_transaction(self, tx):

        for peer in self.peer_manager.get_peers():
            try:
                requests.post(f"{peer}/receive_transaction", json=tx.to_dict())
            except Exception as e:
                logger.warning(f"Failed to send tx to {peer}: {e}")

    def broadcast_block(self, block):

        for peer in self.peer_manager.get_peers():
            try:
                requests.post(f"{peer}/receive_block", json={
                    "index": block.index,
                    "transactions": [t.to_dict() for t in block.transactions],
                    "previous_hash": block.previous_hash,
                    "nonce": block.nonce,
                    "hash": block.hash
                })
            except Exception as e:
                logger.warning(f"Failed to send block to {peer}: {e}")


    def connect_to_network(self, bootstrap_node):
        try:
            response = requests.post(
                f"{bootstrap_node}/register_peer",
                json={"address": self.address}
            )

            data = response.json()

            for peer in data["peers"]:
                if peer != self.address:
                    self.peer_manager.add_peer(peer)

            # also add bootstrap itself
            self.peer_manager.add_peer(bootstrap_node)

            logger.info(f"Connected to network via {bootstrap_node}")

        except Exception as e:
            logger.error(f"Connection failed: {e}")