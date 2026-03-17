class PeerManager:

    def __init__(self):
        self.peers = set()

    def add_peer(self, address):
        self.peers.add(address)

    def remove_peer(self, address):
        self.peers.discard(address)

    def get_peers(self):
        return list(self.peers)