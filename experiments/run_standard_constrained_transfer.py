from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from omnisafe import Agent

ROOT = Path(__file__).resolve().parents[1]


def _run_once(algo: str, env_id: str, seed: int, total_steps: int, steps_per_epoch: int) -> dict:
    agent = Agent(
        algo,
        env_id,
        custom_cfgs={
            "seed": seed,
            "train_cfgs": {"total_steps": total_steps, "parallel": 1, "vector_env_nums": 1, "device": "cpu"},
            "algo_cfgs": {"steps_per_epoch": steps_per_epoch},
            "logger_cfgs": {"use_wandb": False},
        },
    )
    ep_ret, ep_cost, ep_len = agent.learn()
    return {
        "final_ep_return": float(ep_ret),
        "final_ep_cost": float(ep_cost),
        "final_ep_length": float(ep_len),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard constrained transfer benchmarks.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seed list for multi-seed transfer runs.")
    parser.add_argument("--total-steps", type=int, default=3000, help="Total training steps per run.")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Steps per epoch in OmniSafe config.")
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["SafetyPointGoal1-v0", "SafetyHalfCheetahVelocity-v1"],
        help="Environment IDs to run.",
    )
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        default=["CPO", "P3O", "PPOLag"],
        help="OmniSafe constrained algorithms to run.",
    )
    return parser.parse_args()


def _aggregate_seed_runs(seed_runs: list[dict]) -> dict:
    ret_vals = [r["final_ep_return"] for r in seed_runs]
    cost_vals = [r["final_ep_cost"] for r in seed_runs]
    len_vals = [r["final_ep_length"] for r in seed_runs]

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        finite_vals = [float(v) for v in vals if math.isfinite(float(v))]
        if not finite_vals:
            return float("nan"), float("nan")
        if len(finite_vals) == 1:
            return finite_vals[0], 0.0
        return statistics.mean(finite_vals), statistics.pstdev(finite_vals)

    ret_mean, ret_std = _mean_std(ret_vals)
    cost_mean, cost_std = _mean_std(cost_vals)
    len_mean, len_std = _mean_std(len_vals)

    return {
        "n_seeds": len(seed_runs),
        "seed_runs": seed_runs,
        "final_ep_return_mean": float(ret_mean),
        "final_ep_return_std": float(ret_std),
        "final_ep_cost_mean": float(cost_mean),
        "final_ep_cost_std": float(cost_std),
        "final_ep_length_mean": float(len_mean),
        "final_ep_length_std": float(len_std),
    }


def main() -> int:
    args = _parse_args()
    envs = args.envs
    algos = args.algos

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "config": {
            "seeds": args.seeds,
            "total_steps": args.total_steps,
            "steps_per_epoch": args.steps_per_epoch,
            "envs": envs,
            "algos": algos,
        },
        "results": {},
        "errors": {},
        "notes": {},
    }
    has_issue = False
    has_success = False

    for env_id in envs:
        payload["results"][env_id] = {}
        for algo in algos:
            seed_runs: list[dict] = []
            for seed in args.seeds:
                try:
                    out = _run_once(algo, env_id, seed, args.total_steps, args.steps_per_epoch)
                    out["seed"] = int(seed)
                    seed_runs.append(out)
                    has_success = True
                    vals = [out["final_ep_return"], out["final_ep_cost"], out["final_ep_length"]]
                    if not all(math.isfinite(v) for v in vals):
                        has_issue = True
                        payload["notes"].setdefault(env_id, {}).setdefault(algo, []).append(
                            f"seed {seed}: non-finite episode metric(s)"
                        )
                except Exception as exc:  # noqa: BLE001
                    has_issue = True
                    payload["errors"].setdefault(env_id, {}).setdefault(algo, []).append(
                        f"seed {seed}: {exc}"
                    )

            if seed_runs:
                payload["results"][env_id][algo] = _aggregate_seed_runs(seed_runs)

    if has_issue and has_success:
        payload["status"] = "unstable"
    elif has_issue and not has_success:
        payload["status"] = "blocked"

    out_json = ROOT / "experiments" / "results" / "standard_constrained_transfer.json"
    out_md = ROOT / "docs" / "STANDARD_CONSTRAINED_TRANSFER.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Standard Constrained Transfer",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        f"Status: {payload['status']}",
        f"Seeds: {payload['config']['seeds']}",
        f"Total steps per run: {payload['config']['total_steps']}",
        f"Steps per epoch: {payload['config']['steps_per_epoch']}",
        f"Environments: {payload['config']['envs']}",
        f"Algorithms: {payload['config']['algos']}",
        "",
    ]

    if payload["status"] in {"blocked", "unstable"}:
        lines.append("## Issues")
        for env_id, errmap in payload["errors"].items():
            lines.append(f"- {env_id}")
            for algo, msg_list in errmap.items():
                for msg in msg_list:
                    lines.append(f"  - {algo}: {msg}")
        for env_id, notemap in payload["notes"].items():
            lines.append(f"- {env_id}")
            for algo, msg_list in notemap.items():
                for msg in msg_list:
                    lines.append(f"  - {algo}: {msg}")
        lines.append("")

    if payload["results"]:
        lines.append("## Results")
        for env_id, by_algo in payload["results"].items():
            if not by_algo:
                continue
            lines.append(f"### {env_id}")
            lines.append("| Algo | Seeds | Return mean±std | Cost mean±std | EpLen mean±std |")
            lines.append("|---|---:|---:|---:|---:|")
            for algo, vals in by_algo.items():
                lines.append(
                    f"| {algo} | {vals['n_seeds']} | {vals['final_ep_return_mean']:.2f} +- {vals['final_ep_return_std']:.2f} | {vals['final_ep_cost_mean']:.2f} +- {vals['final_ep_cost_std']:.2f} | {vals['final_ep_length_mean']:.2f} +- {vals['final_ep_length_std']:.2f} |"
                )
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
