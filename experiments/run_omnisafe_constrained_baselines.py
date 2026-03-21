from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from omnisafe import Agent

from ai.rl import omnisafe_envs  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OmniSafeConfig:
    seeds: tuple[int, ...]
    total_steps: int
    episode_steps_scaled: int
    episode_steps_defi: int
    adversarial_intensity: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=str,
        default=(
            "3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,"
            "43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,"
            "83,85,87,89,91,93,95,97,99,101"
        ),
    )
    p.add_argument("--total-steps", type=int, default=24000)
    p.add_argument("--episode-steps-scaled", type=int, default=140)
    p.add_argument("--episode-steps-defi", type=int, default=120)
    p.add_argument("--adversarial-intensity", type=float, default=0.7)
    p.add_argument(
        "--algos",
        type=str,
        default="CPO,P3O,PPOLag,SACLag,FOCOPS",
        help="Comma-separated OmniSafe algorithm names.",
    )
    return p.parse_args()


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def permutation_p_value(a: list[float], b: list[float], n_shuffles: int = 20000, seed: int = 0) -> float:
    rng = random.Random(seed)
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pooled = a + b
    m = len(a)
    ge_count = 0
    for _ in range(n_shuffles):
        rng.shuffle(pooled)
        x = pooled[:m]
        y = pooled[m:]
        diff = abs(statistics.mean(x) - statistics.mean(y))
        if diff >= obs - 1e-12:
            ge_count += 1
    return float((ge_count + 1) / (n_shuffles + 1))


def significance_marker(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def build_custom_cfgs(seed: int, env_name: str, cfg: OmniSafeConfig) -> dict:
    del env_name
    return {
        "seed": seed,
        "train_cfgs": {
            "device": "cpu",
            "parallel": 1,
            "vector_env_nums": 1,
            "total_steps": cfg.total_steps,
        },
        "algo_cfgs": {
            "steps_per_epoch": min(500, cfg.total_steps),
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 1,
        },
    }


def run_official_baseline(algo: str, env_id: str, env_name: str, cfg: OmniSafeConfig, seed: int) -> dict[str, float]:
    os.environ["QAI_OMNISAFE_ADVERSARIAL_INTENSITY"] = str(cfg.adversarial_intensity)
    if env_name == "scaled":
        os.environ["QAI_OMNISAFE_EPISODE_STEPS_SCALED"] = str(cfg.episode_steps_scaled)
    else:
        os.environ["QAI_OMNISAFE_EPISODE_STEPS_DEFI"] = str(cfg.episode_steps_defi)
    custom_cfgs = build_custom_cfgs(seed, env_name, cfg)
    agent = Agent(algo, env_id, custom_cfgs=custom_cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()
    return {
        "final_ep_return": float(ep_ret),
        "final_ep_cost": float(ep_cost),
        "final_ep_length": float(ep_len),
    }


def main() -> int:
    args = parse_args()
    cfg = OmniSafeConfig(
        seeds=tuple(int(x) for x in args.seeds.split(",") if x.strip()),
        total_steps=args.total_steps,
        episode_steps_scaled=args.episode_steps_scaled,
        episode_steps_defi=args.episode_steps_defi,
        adversarial_intensity=args.adversarial_intensity,
    )

    env_map = {
        "scaled": "QAIChainScaled-v0",
        "defi": "QAIChainDeFi-v0",
    }
    requested_algos = [x.strip() for x in args.algos.split(",") if x.strip()]
    alias_map = {
        "CPO": "cpo_official",
        "P3O": "p3o_official",
        "PPOLag": "ppo_lagrangian_official",
        "SACLag": "sac_lagrangian_official",
        "FOCOPS": "focops_official",
    }
    invalid = [a for a in requested_algos if a not in alias_map]
    if invalid:
        raise ValueError(f"Unsupported algorithms: {invalid}. Allowed: {list(alias_map)}")
    algos = {alias_map[name]: name for name in requested_algos}

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "omnisafe_official",
        "config": {
            "seeds": list(cfg.seeds),
            "total_steps": cfg.total_steps,
            "episode_steps_scaled": cfg.episode_steps_scaled,
            "episode_steps_defi": cfg.episode_steps_defi,
            "adversarial_intensity": cfg.adversarial_intensity,
            "algorithms": algos,
            "environments": env_map,
        },
        "results": {},
    }

    for env_name, env_id in env_map.items():
        payload["results"][env_name] = {}
        reward_by_alias: dict[str, list[float]] = {}
        for alias, algo_name in algos.items():
            per_seed = [run_official_baseline(algo_name, env_id, env_name, cfg, s) for s in cfg.seeds]
            reward_vals = [x["final_ep_return"] for x in per_seed]
            reward_by_alias[alias] = reward_vals
            payload["results"][env_name][alias] = {
                "algo": algo_name,
                "per_seed": per_seed,
                "aggregate": {
                    "final_ep_return": aggregate(reward_vals),
                    "final_ep_cost": aggregate([x["final_ep_cost"] for x in per_seed]),
                    "final_ep_length": aggregate([x["final_ep_length"] for x in per_seed]),
                },
            }

        best_alias = max(
            reward_by_alias,
            key=lambda k: payload["results"][env_name][k]["aggregate"]["final_ep_return"]["mean"],
        )
        for alias in algos:
            if alias == best_alias:
                p_val = 1.0
                marker = "---"
            else:
                p_val = permutation_p_value(reward_by_alias[alias], reward_by_alias[best_alias], n_shuffles=20000, seed=42)
                marker = significance_marker(p_val)
            payload["results"][env_name][alias]["vs_best_return"] = {
                "best_alias": best_alias,
                "p_value": p_val,
                "significance": marker,
            }

    out_json = ROOT / "experiments" / "results" / "omnisafe_constrained_baseline_suite.json"
    out_md = ROOT / "docs" / "OMNISAFE_CONSTRAINED_BASELINE_SUITE.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# OmniSafe Official Constrained Baselines",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        f"Algorithms: {', '.join(requested_algos)} (official OmniSafe implementations)",
        "",
    ]
    for env_name in env_map:
        lines.append(f"## {env_name.upper()} Environment")
        lines.append("| Baseline | Final Return (mean±std) | Final Cost (mean±std) | Final Ep Len (mean±std) | p vs best return |")
        lines.append("|---|---:|---:|---:|---:|")
        for alias, algo_name in algos.items():
            a = payload["results"][env_name][alias]["aggregate"]
            v = payload["results"][env_name][alias]["vs_best_return"]
            p_text = "---" if v["significance"] == "---" else f"{v['p_value']:.3f} ({v['significance']})"
            lines.append(
                f"| {algo_name} | {a['final_ep_return']['mean']:.2f} ± {a['final_ep_return']['std']:.2f} | "
                f"{a['final_ep_cost']['mean']:.3f} ± {a['final_ep_cost']['std']:.3f} | "
                f"{a['final_ep_length']['mean']:.2f} ± {a['final_ep_length']['std']:.2f} | {p_text} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote OmniSafe constrained baseline JSON: {out_json}")
    print(f"Wrote OmniSafe constrained baseline markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())