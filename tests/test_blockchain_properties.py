import random
from typing import Any, cast

from core.blockchain.block import Block
from core.blockchain.blockchain import Blockchain
from core.blockchain.mempool import Mempool
from core.blockchain.transaction import Transaction
from core.blockchain.miner import proof_of_work


def _valid_tx(idx: int) -> Transaction:
    return Transaction(
        sender=f"pub_{idx}",
        receiver=f"recv_{idx}",
        amount=1.0 + idx,
        signature="sig",
    )


def test_property_chain_links_and_pow_over_randomized_sequences():
    rng = random.Random(1337)
    chain = Blockchain(difficulty=1)
    chain_dyn = cast(Any, chain)
    chain_dyn.validate_transaction = lambda tx: True
    mempool = Mempool()

    for _ in range(15):
        tx_count = rng.randint(1, 5)
        for i in range(tx_count):
            mempool.add_transaction(_valid_tx(i))

        block = Block(
            index=len(chain.chain),
            transactions=mempool.get_transactions(),
            previous_hash=chain.last_block().hash,
            zk_proof="proof",
        )
        mined = proof_of_work(block, chain.difficulty)

        assert chain.is_valid_block(mined, chain.last_block())
        chain.add_block(mined)

    for i in range(1, len(chain.chain)):
        prev_block = chain.chain[i - 1]
        current = chain.chain[i]
        assert current.previous_hash == prev_block.hash
        assert current.hash.startswith("0" * chain.difficulty)
        assert current.compute_hash() == current.hash


def test_property_tampering_breaks_validation():
    chain = Blockchain(difficulty=1)
    chain_dyn = cast(Any, chain)
    chain_dyn.validate_transaction = lambda tx: True
    block = Block(
        index=1,
        transactions=[_valid_tx(1)],
        previous_hash=chain.last_block().hash,
        zk_proof="proof",
    )
    mined = proof_of_work(block, chain.difficulty)
    assert chain.is_valid_block(mined, chain.last_block())

    mined.previous_hash = "tampered"
    assert not chain.is_valid_block(mined, chain.last_block())
