from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.blockchain.blockchain import Blockchain
from core.blockchain.mempool import Mempool
from network.p2p_node import P2PNode
from network.peer_manager import PeerManager
from network.rpc_server import app, init_node


def main() -> int:
    blockchain = Blockchain(difficulty=1)
    mempool = Mempool()
    peers = PeerManager()
    p2p = P2PNode("http://127.0.0.1:8000", peers)
    init_node(blockchain, mempool, p2p)

    client = TestClient(app)
    response = client.get("/openapi.json")
    response.raise_for_status()
    openapi = response.json()

    out_json = ROOT / "docs" / "api_openapi.json"
    out_json.write_text(json.dumps(openapi, indent=2), encoding="utf-8")

    paths = openapi.get("paths", {})
    lines = [
        "# API Schema",
        "",
        "This document is generated from FastAPI OpenAPI output.",
        "",
        "## Endpoints",
        "",
    ]
    for path, methods in sorted(paths.items()):
        method_list = ", ".join(sorted(m.upper() for m in methods.keys()))
        lines.append(f"- {method_list} {path}")

    lines.extend(
        [
            "",
            "## Schema Artifact",
            "",
            "- JSON schema: docs/api_openapi.json",
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python scripts/generate_api_schema_docs.py",
            "```",
        ]
    )

    out_md = ROOT / "docs" / "API_SCHEMA.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote API schema JSON: {out_json}")
    print(f"Wrote API schema markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
