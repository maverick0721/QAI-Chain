from fastapi.testclient import TestClient

from core.blockchain.blockchain import Blockchain
from core.blockchain.mempool import Mempool
from network.p2p_node import P2PNode
from network.peer_manager import PeerManager
from network.rpc_server import app, init_node


def _client() -> TestClient:
    blockchain = Blockchain(difficulty=1)
    mempool = Mempool()
    peer_manager = PeerManager()
    p2p = P2PNode("http://127.0.0.1:9010", peer_manager)
    init_node(blockchain, mempool, p2p)
    return TestClient(app)


def test_openapi_includes_critical_paths():
    client = _client()
    response = client.get("/openapi.json")
    assert response.status_code == 200

    paths = response.json()["paths"]
    assert "/register_peer" in paths
    assert "/receive_transaction" in paths
    assert "/mine" in paths
    assert "/metrics" in paths
    assert "/healthz" in paths


def test_register_peer_rejects_invalid_payload():
    client = _client()
    response = client.post("/register_peer", json={})
    assert response.status_code == 422


def test_receive_transaction_rejects_negative_amount():
    client = _client()
    response = client.post(
        "/receive_transaction",
        json={
            "sender": "alice",
            "receiver": "bob",
            "amount": -5,
            "signature": "sig",
        },
    )
    assert response.status_code == 422


def test_transaction_endpoint_rejects_missing_receiver():
    client = _client()
    response = client.post("/transaction", json={"amount": 5})
    assert response.status_code == 422


def test_metrics_and_trace_headers_present():
    client = _client()
    response = client.get("/healthz", headers={"x-trace-id": "trace-123"})
    assert response.status_code == 200
    assert response.headers.get("x-trace-id") == "trace-123"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["requests_total"] >= 1
    assert "/healthz" in payload["requests_by_path"]