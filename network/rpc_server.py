from fastapi import FastAPI

from core.blockchain.transaction import Transaction
from core.blockchain.block import Block
from core.blockchain.miner import proof_of_work

from crypto.pqc.keypair import PQCKeypair
from crypto.pqc.transaction_signer import sign_transaction

from ai.integration.blockchain_ai_bridge import AIBridge

app = FastAPI()

node = None



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



# Peer Discovery

@app.post("/register_peer")
def register_peer(data: dict):

    new_peer = data["address"]

    node["p2p"].peer_manager.add_peer(new_peer)

    return {
        "peers": node["p2p"].peer_manager.get_peers()
    }


@app.get("/peers")
def get_peers():

    return {
        "peers": node["p2p"].peer_manager.get_peers()
    }



# Transaction Endpoint (PQC SIGNED)

@app.post("/transaction")
def add_transaction(tx_data: dict):

    # Create wallet for this node
    wallet = PQCKeypair()

    tx = Transaction(
        sender=wallet.get_public_key(),
        receiver=tx_data["receiver"],
        amount=tx_data["amount"]
    )

    # Sign transaction
    tx = sign_transaction(tx, wallet.private_key)

    node["mempool"].add_transaction(tx)

    # Broadcast
    node["p2p"].broadcast_transaction(tx)

    return {
        "status": "transaction created & broadcasted",
        "sender": tx.sender
    }



# Receive Transaction (from peers)

@app.post("/receive_transaction")
def receive_transaction(tx_data: dict):

    tx = Transaction(
        tx_data["sender"],
        tx_data["receiver"],
        tx_data["amount"],
        tx_data.get("signature")
    )

    node["mempool"].add_transaction(tx)

    return {"status": "transaction received"}


# Receive Block

@app.post("/block")
def receive_block(block_data: dict):

    blockchain = node["blockchain"]

    block = Block(
        index=block_data["index"],
        transactions=[
            Transaction(**t) for t in block_data["transactions"]
        ],
        previous_hash=block_data["previous_hash"]
    )

    block.nonce = block_data["nonce"]
    block.hash = block_data["hash"]

    if blockchain.is_valid_block(block, blockchain.last_block()):
        blockchain.add_block(block)
        return {"status": "block added"}

    return {"status": "invalid block"}


# Get Chain

@app.get("/chain")
def get_chain():

    return [
        {
            "index": b.index,
            "hash": b.hash,
            "tx": len(b.transactions)
        }
        for b in node["blockchain"].chain
    ]


# Mine Block (AI + PQC SYSTEM)

@app.post("/mine")
def mine():

    blockchain = node["blockchain"]
    mempool = node["mempool"]

    transactions = mempool.get_transactions()

    if not transactions:
        return {"status": "no transactions"}

    
    # AI decision
    
    decision = node["ai"].decide()

    print("🧠 AI Decision:", decision)

    
    # Create block
    
    new_block = Block(
        index=len(blockchain.chain),
        transactions=transactions,
        previous_hash=blockchain.last_block().hash
    )

    mined_block = proof_of_work(new_block, blockchain.difficulty)

    blockchain.add_block(mined_block)

    # Broadcast block
    node["p2p"].broadcast_block(mined_block)

    return {
        "status": "block mined",
        "hash": mined_block.hash
    }