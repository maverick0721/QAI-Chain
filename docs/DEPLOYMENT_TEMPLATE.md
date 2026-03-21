# Deployment Template

## Single Node

```bash
cd deploy
docker compose -f docker-compose.single-node.yml up --build
```

RPC endpoint:

- http://127.0.0.1:8000

## Local Testnet (3 Nodes)

```bash
cd deploy
docker compose -f docker-compose.local-testnet.yml up --build
```

Node endpoints:

- node1: http://127.0.0.1:8001
- node2: http://127.0.0.1:8002
- node3: http://127.0.0.1:8003

## Smoke Validation

```bash
curl http://127.0.0.1:8001/healthz
curl http://127.0.0.1:8001/metrics
```
