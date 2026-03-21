# QAI-Chain Architecture

## System Goal

QAI-Chain explores a hybrid stack where secure decentralized infrastructure and adaptive intelligence co-evolve:

- blockchain provides verifiable state progression
- post-quantum cryptography provides transaction integrity
- AI agents provide adaptive governance signals
- quantum models provide experimentation paths for richer representation learning

## Layered Breakdown

1. Network Layer
- [network/p2p_node.py](network/p2p_node.py): peer registration and gossip-style propagation for txs and blocks
- [network/rpc_server.py](network/rpc_server.py): FastAPI endpoints for chain interaction

2. Consensus/Core Chain Layer
- [core/blockchain/blockchain.py](core/blockchain/blockchain.py): chain state and validation logic
- [core/blockchain/miner.py](core/blockchain/miner.py): proof-of-work mining loop
- [core/blockchain/mempool.py](core/blockchain/mempool.py): transaction staging area

3. Cryptography Layer
- [crypto/pqc/keypair.py](crypto/pqc/keypair.py): PQC key generation
- [crypto/pqc/transaction_signer.py](crypto/pqc/transaction_signer.py): signing pipeline
- [crypto/pqc/verification.py](crypto/pqc/verification.py): signature verification path

4. AI Governance Layer
- [ai/data/blockchain_dataset.py](ai/data/blockchain_dataset.py): chain-to-feature extraction
- [ai/models](ai/models): encoder/policy/value models
- [ai/rl](ai/rl): environment, PPO agent, and training loop

5. Quantum Layer
- [quantum/models/qnn.py](quantum/models/qnn.py): QNN model path
- [quantum/transformer/q_transformer.py](quantum/transformer/q_transformer.py): hybrid transformer with quantum attention
- [quantum/kernels/quantum_kernel.py](quantum/kernels/quantum_kernel.py): kernel-based quantum similarity

6. ZK Integration Layer
- [zk/integration/ai_verification.py](zk/integration/ai_verification.py): integration points for proof checks
- [zk/circuits/ai_inference.circom](zk/circuits/ai_inference.circom): proof circuit scaffold

## Execution Flow (Typical)

1. Node boots via [scripts/run_node.py](scripts/run_node.py)
2. Transactions are created/signed and added to mempool
3. Mining path builds candidate block and performs PoW
4. AI bridge generates governance signal during block production
5. Block is broadcast to peers and validated by chain rules

## Current Engineering Guardrails

- automated tests via pytest
- healthcheck script for syntax/import/runtime smoke
- CI workflow to enforce checks on push and PR

## Suggested Next Hardening Steps

- add contract tests for RPC endpoints
- add deterministic fixtures for chain/network integration
- formalize benchmark baselines in versioned results files