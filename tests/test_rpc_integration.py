from fastapi.testclient import TestClient

from core.blockchain.blockchain import Blockchain
from core.blockchain.mempool import Mempool
from network.p2p_node import P2PNode
from network.peer_manager import PeerManager
from network.rpc_server import app, init_node


def _client_with_node(difficulty: int = 1) -> TestClient:
    blockchain = Blockchain(difficulty=difficulty)
    mempool = Mempool()
    peers = PeerManager()
    p2p = P2PNode("http://127.0.0.1:9000", peers)
    init_node(blockchain, mempool, p2p)
    return TestClient(app)


def test_register_peer_and_list_peers():
    client = _client_with_node()

    response = client.post("/register_peer", json={"address": "http://peer-1:8000"})
    assert response.status_code == 200

    peers = response.json()["peers"]
    assert "http://peer-1:8000" in peers

    response = client.get("/peers")
    assert response.status_code == 200
    assert "http://peer-1:8000" in response.json()["peers"]


def test_receive_transaction_and_mine_block_flow():
    client = _client_with_node(difficulty=1)

    tx_payload = {
        "sender": "alice_pubkey",
        "receiver": "bob_pubkey",
        "amount": 5,
        "signature": "dummy_signature",
    }

    response = client.post("/receive_transaction", json=tx_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "transaction received"

    response = client.post("/mine")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "block mined"
    assert body["zk_proof"] == "zk_proof_placeholder"

    chain = client.get("/chain")
    assert chain.status_code == 200
    chain_body = chain.json()
    assert len(chain_body) == 2
    assert chain_body[-1]["tx"] == 1


def test_mine_without_transactions_returns_no_transactions_status():
    client = _client_with_node(difficulty=1)

    response = client.post("/mine")
    assert response.status_code == 200
    assert response.json()["status"] == "no transactions"