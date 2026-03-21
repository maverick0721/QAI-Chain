from fastapi import FastAPI
from typing import Any
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from core.blockchain.transaction import Transaction
from core.blockchain.block import Block
from core.blockchain.miner import proof_of_work

from crypto.pqc.keypair import PQCKeypair
from crypto.pqc.transaction_signer import sign_transaction

from ai.integration.blockchain_ai_bridge import AIBridge
from network.schemas import (
    BlockReceiveRequest,
    BlockResponse,
    ChainBlockResponse,
    MineResponse,
    PeerListResponse,
    PeerRegistrationRequest,
    StatusResponse,
    TransactionCreateRequest,
    TransactionCreatedResponse,
    TransactionReceiveRequest,
)

app = FastAPI()

node: dict[str, Any] | None = None
request_metrics: dict[str, Any] = {
    "total_requests": 0,
    "by_path": {},
    "by_status": {},
    "latency_ms_sum": 0.0,
    "latency_ms_max": 0.0,
}


def _require_node() -> dict[str, Any]:
    if node is None:
        raise RuntimeError("Node is not initialized. Call init_node first.")
    return node


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    start = time.perf_counter()
    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    request_metrics["total_requests"] += 1
    path = request.url.path
    request_metrics["by_path"][path] = request_metrics["by_path"].get(path, 0) + 1
    code = str(response.status_code)
    request_metrics["by_status"][code] = request_metrics["by_status"].get(code, 0) + 1
    request_metrics["latency_ms_sum"] += elapsed_ms
    request_metrics["latency_ms_max"] = max(request_metrics["latency_ms_max"], elapsed_ms)

    response.headers["x-trace-id"] = trace_id
    response.headers["x-response-time-ms"] = f"{elapsed_ms:.3f}"
    return response



# Initialize node

def init_node(blockchain, mempool, p2p):
    global node

    ai_bridge = AIBridge(blockchain)

    node = {
        "blockchain": blockchain,
        "mempool": mempool,
        "p2p": p2p,
        "ai": ai_bridge
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "initialized": node is not None}


@app.get("/metrics")
def metrics():
    total = request_metrics["total_requests"]
    avg_latency = request_metrics["latency_ms_sum"] / total if total else 0.0
    return JSONResponse(
        {
            "requests_total": total,
            "requests_by_path": request_metrics["by_path"],
            "requests_by_status": request_metrics["by_status"],
            "latency_ms_avg": avg_latency,
            "latency_ms_max": request_metrics["latency_ms_max"],
        }
    )



# Peer Discovery

@app.post("/register_peer", response_model=PeerListResponse)
def register_peer(data: PeerRegistrationRequest):

    n = _require_node()

    new_peer = data.address

    n["p2p"].peer_manager.add_peer(new_peer)

    return {
        "peers": n["p2p"].peer_manager.get_peers()
    }


@app.get("/peers", response_model=PeerListResponse)
def get_peers():

    n = _require_node()

    return {
        "peers": n["p2p"].peer_manager.get_peers()
    }



# Transaction Endpoint (PQC SIGNED)

@app.post("/transaction", response_model=TransactionCreatedResponse)
def add_transaction(tx_data: TransactionCreateRequest):

    n = _require_node()

    # Create wallet for this node
    wallet = PQCKeypair()

    tx = Transaction(
        sender=wallet.get_public_key(),
        receiver=tx_data.receiver,
        amount=tx_data.amount
    )

    # Sign transaction
    tx = sign_transaction(tx, wallet.private_key)

    n["mempool"].add_transaction(tx)

    # Broadcast
    n["p2p"].broadcast_transaction(tx)

    return {
        "status": "transaction created & broadcasted",
        "sender": tx.sender
    }



# Receive Transaction (from peers)

@app.post("/receive_transaction", response_model=StatusResponse)
def receive_transaction(tx_data: TransactionReceiveRequest):

    n = _require_node()

    tx = Transaction(
        tx_data.sender,
        tx_data.receiver,
        tx_data.amount,
        tx_data.signature,
    )

    n["mempool"].add_transaction(tx)

    return {"status": "transaction received"}


# Receive Block

@app.post("/block", response_model=BlockResponse)
def receive_block(block_data: BlockReceiveRequest):

    n = _require_node()

    blockchain = n["blockchain"]

    block = Block(
        index=block_data.index,
        transactions=[
            Transaction(**t.model_dump()) for t in block_data.transactions
        ],
        previous_hash=block_data.previous_hash,
        zk_proof=block_data.zk_proof,

    )

    block.nonce = block_data.nonce
    block.hash = block_data.hash

    if blockchain.is_valid_block(block, blockchain.last_block()):
        blockchain.add_block(block)
        return {"status": "block added"}

    return {"status": "invalid block"}


# Get Chain

@app.get("/chain", response_model=list[ChainBlockResponse])
def get_chain():

    n = _require_node()

    return [
        {
            "index": b.index,
            "hash": b.hash,
            "tx": len(b.transactions)
        }
        for b in n["blockchain"].chain
    ]


# Mine Block (AI + ZK + PQC SYSTEM)

@app.post("/mine", response_model=MineResponse)
def mine():

    n = _require_node()

    blockchain = n["blockchain"]
    mempool = n["mempool"]

    transactions = mempool.get_transactions()

    if not transactions:
        return {"status": "no transactions"}

    
    # AI decision
    
    decision = n["ai"].decide()
    print("AI Decision:", decision)

    
    # ZK Proof (SIMULATED FOR NOW)
    
    zk_proof = "zk_proof_placeholder"

    
    # Create block (WITH ZK PROOF)
    
    new_block = Block(
        index=len(blockchain.chain),
        transactions=transactions,
        previous_hash=blockchain.last_block().hash,
        zk_proof=zk_proof   # NEW
    )

    mined_block = proof_of_work(new_block, blockchain.difficulty)

    blockchain.add_block(mined_block)

    # Broadcast block
    n["p2p"].broadcast_block(mined_block)

    return {
        "status": "block mined",
        "hash": mined_block.hash,
        "zk_proof": zk_proof
    }