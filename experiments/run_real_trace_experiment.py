from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from experiments.run_constrained_baseline_suite import Config, aggregate, run_method

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "experiments" / "data"
TRACE_RAW = TRACE_DIR / "ethereum_gasprice_daily_raw.csv"
TRACE_FEATURES = TRACE_DIR / "ethereum_gasprice_trace_features.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="3,5,7,9,11")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--intensity", type=float, default=0.7)
    p.add_argument("--uncertainty-k", type=int, default=4)
    p.add_argument("--uncertainty-every", type=int, default=5)
    return p.parse_args()


def download_etherscan_gas_csv(out_path: Path) -> None:
    url = "https://etherscan.io/chart/gasprice?output=csv"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def build_trace_features(raw_csv: Path, out_csv: Path) -> None:
    raw = pd.read_csv(raw_csv)
    wei = raw["Value (Wei)"].astype(float).clip(lower=0.0)
    gwei = wei / 1e9
    s = pd.Series(gwei)
    roll = s.rolling(14, min_periods=2)
    z = (s - roll.mean()) / (roll.std().replace(0, np.nan))
    z = z.fillna(0.0).clip(-3.0, 3.0)

    intensity = ((z + 3.0) / 6.0).clip(0.0, 1.0)
    delta = s.diff().fillna(0.0)
    vol = s.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).abs().clip(0.0, 1.5)

    feat = pd.DataFrame(
        {
            "date_utc": raw["Date(UTC)"],
            "gas_gwei": s,
            "intensity": intensity,
            "demand_base": 8.0 + 3.5 * intensity,
            "demand_std": 0.8 + 0.8 * vol,
            "attack_drift": 0.12 + 0.20 * intensity,
            "attack_noise": 0.05 + 0.06 * vol,
            "volatility_drift": 0.015 + 0.03 * intensity,
            "volatility_noise": 0.02 + 0.04 * vol,
            "mev_drift": 0.18 + 0.25 * intensity,
            "mev_noise": 0.03 + 0.05 * vol,
            "oracle_drift": 0.03 + 0.04 * intensity,
            "oracle_noise": 0.008 + 0.01 * vol,
            "swap_flow_std": 0.20 + 0.20 * vol,
            "gas_delta_gwei": delta,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_csv, index=False)


def main() -> int:
    args = parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())

    download_etherscan_gas_csv(TRACE_RAW)
    build_trace_features(TRACE_RAW, TRACE_FEATURES)

    cfg = Config(
        seeds=seeds,
        episodes=args.episodes,
        steps=args.steps,
        intensity=args.intensity,
        uncertainty_k=args.uncertainty_k,
        uncertainty_every=args.uncertainty_every,
        trace_csv_path=str(TRACE_FEATURES),
    )

    methods = ["heuristic", "ppo_lagrangian", "robust_qgate_shield"]
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_source": "etherscan_gasprice_daily",
        "trace_raw_csv": str(TRACE_RAW),
        "trace_features_csv": str(TRACE_FEATURES),
        "config": {
            "seeds": list(seeds),
            "episodes": cfg.episodes,
            "steps": cfg.steps,
            "uncertainty_k": cfg.uncertainty_k,
            "uncertainty_every": cfg.uncertainty_every,
        },
        "results": {},
    }

    for env_name in ["scaled", "defi"]:
        payload["results"][env_name] = {}
        for method in methods:
            per_seed = [run_method(method, env_name, cfg, s) for s in seeds]
            payload["results"][env_name][method] = {
                "per_seed": per_seed,
                "aggregate": aggregate(per_seed),
            }

    out_json = ROOT / "experiments" / "results" / "real_trace_replay_results.json"
    out_md = ROOT / "docs" / "REAL_TRACE_EXPERIMENT.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    names = {
        "heuristic": "Heuristic",
        "ppo_lagrangian": "PPO-Lagrangian",
        "robust_qgate_shield": "Robust QGate+Shield",
    }
    lines = [
        "# Real-Trace Replay Experiment",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "Trace source: Etherscan Ethereum daily gas price history",
        "",
    ]
    for env_name in ["scaled", "defi"]:
        lines.append(f"## {env_name.upper()} Environment")
        lines.append("| Method | Reward (mean±std) | Violations | Target MAE | Fallback | Latency |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method in methods:
            a = payload["results"][env_name][method]["aggregate"]
            lines.append(
                f"| {names[method]} | {a['reward_mean_mean']:.2f} ± {a['reward_mean_std']:.2f} | "
                f"{a['violations_mean_mean']:.2f} | {a['target_error_mean_mean']:.3f} | "
                f"{a['fallback_rate_mean_mean']:.3f} | {a['latency_mean_mean']:.3f} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote trace replay JSON: {out_json}")
    print(f"Wrote trace replay markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())