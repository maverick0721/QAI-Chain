from pathlib import Path
import sys

# Allow running this script directly from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from core.blockchain.blockchain import Blockchain
from core.blockchain.mempool import Mempool
from network.peer_manager import PeerManager
from network.p2p_node import P2PNode
from network.rpc_server import app, init_node


def main():

  
    # Parse arguments
  
    bootstrap = None
    port = 8000

    if len(sys.argv) > 1:
        bootstrap = sys.argv[1]

    if len(sys.argv) > 2:
        port = int(sys.argv[2])

    node_address = f"http://127.0.0.1:{port}"

  
    # Initialize components
  
    blockchain = Blockchain()
    mempool = Mempool()
    peers = PeerManager()

    p2p = P2PNode(node_address, peers)

    # Inject into FastAPI server
    init_node(blockchain, mempool, p2p)

  
    # Auto-connect to network
  
    if bootstrap:
        p2p.connect_to_network(bootstrap)

  
    # Start server
  
    print(f"🚀 Node running at {node_address}")

    if bootstrap:
        print(f"🔗 Connected to bootstrap node: {bootstrap}")

    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()