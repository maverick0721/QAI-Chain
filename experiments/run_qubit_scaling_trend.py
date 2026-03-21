from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[1]


def _measure_latency_ms(n_qubits: int, n_layers: int, samples: int) -> float:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(x: torch.Tensor, w: torch.Tensor):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(w, wires=range(n_qubits), rotation=qml.RY)
        return qml.expval(qml.PauliZ(0))

    w = 0.02 * torch.randn(n_layers, n_qubits)
    xs = torch.randn(samples, n_qubits)

    # Warmup
    _ = circuit(xs[0], w)

    latencies = []
    for i in range(samples):
        t0 = time.perf_counter()
        _ = circuit(xs[i], w)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate qubit-scaling simulation trend artifacts.")
    parser.add_argument("--qubits", type=str, default="2,4,6,8,10")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    n_qubits_list = [int(x) for x in args.qubits.split(",") if x.strip()]
    n_layers = args.layers

    rows = []
    for n_qubits in n_qubits_list:
        latency_ms = _measure_latency_ms(n_qubits, n_layers, args.samples)
        rows.append(
            {
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "state_dim": int(2**n_qubits),
                "parameter_count": int(n_qubits * n_layers),
                "forward_latency_ms": latency_ms,
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "qubits": n_qubits_list,
            "layers": n_layers,
            "samples": args.samples,
        },
        "results": rows,
    }

    out_json = ROOT / "experiments" / "results" / "qubit_scaling_trend.json"
    out_md = ROOT / "docs" / "QUBIT_SCALING_TREND.md"
    out_fig = ROOT / "paper" / "figures" / "qubit_scaling_trend.pdf"
    out_table = ROOT / "paper" / "tables" / "qubit_scaling_trend.tex"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Qubit Scaling Trend",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "| Qubits | Layers | Parameters | State Dim ($2^n$) | Forward Latency (ms) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['n_qubits']} | {r['n_layers']} | {r['parameter_count']} | {r['state_dim']} | {r['forward_latency_ms']:.4f} |"
        )
    md_lines.append("")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    xs = [r["n_qubits"] for r in rows]
    ys = [r["forward_latency_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_xlabel("Qubit Count")
    ax.set_ylabel("Forward Latency (ms)")
    ax.set_title("Qubit-Scaling Trend (Simulator)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig)
    plt.close(fig)

    table_lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Qubits & Layers & Params & $2^n$ State Dim & Forward Latency (ms) \\",
        r"\midrule",
    ]
    for r in rows:
        table_lines.append(
            f"{r['n_qubits']} & {r['n_layers']} & {r['parameter_count']} & {r['state_dim']} & {r['forward_latency_ms']:.4f} \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_table.write_text("\n".join(table_lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_fig}")
    print(f"Wrote {out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
