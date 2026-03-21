from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results" / "detailed_results.json"
OUT_DIR = ROOT / "paper" / "tables"


def _load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _agg(method_runs: dict[str, dict], key: str) -> np.ndarray:
    return np.array([v[key] for v in method_runs.values()], dtype=np.float64)


def write_trajectory_summary(data: dict) -> None:
    methods = [
        ("random", "Random"),
        ("heuristic_target", "Heuristic"),
        ("ppo_full", "PPO"),
        ("ppo_no_adv_norm", "PPO w/o AdvNorm"),
    ]
    idx = [0, 9, 19, 39, 59]

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Ep1 & Ep10 & Ep20 & Ep40 & Ep60 \\",
        r"\midrule",
    ]

    for key, label in methods:
        traces = _agg(data["methods"][key], "episode_rewards")
        mean_t = traces.mean(axis=0)
        vals = [f"{mean_t[i]:.2f}" for i in idx]
        lines.append(f"{label} & {' & '.join(vals)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "trajectory_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seed_table(data: dict) -> None:
    methods = ["random", "heuristic_target", "ppo_full", "ppo_no_adv_norm"]
    labels = ["Random", "Heuristic", "PPO", "PPO w/o AdvNorm"]
    seeds = sorted(data["methods"]["random"].keys(), key=lambda x: int(x))

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Seed & Random & Heuristic & PPO & PPO w/o AdvNorm \\",
        r"\midrule",
    ]

    for seed in seeds:
        row_vals = []
        for m in methods:
            rewards = np.array(data["methods"][m][seed]["episode_rewards"], dtype=np.float64)
            row_vals.append(f"{rewards[-5:].mean():.2f}")
        lines.append(f"{seed} & {' & '.join(row_vals)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    _ = labels
    (OUT_DIR / "seed_scores.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_difficulty_table(data: dict) -> None:
    methods = [
        ("random", "Random"),
        ("heuristic_target", "Heuristic"),
        ("ppo_full", "PPO"),
        ("ppo_no_adv_norm", "PPO w/o AdvNorm"),
    ]
    target = int(data["config"]["target_difficulty"])

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Mean Final Difficulty & Std Final Difficulty & Mean $|d-d_t|$ \\",
        r"\midrule",
    ]

    for key, label in methods:
        diff = _agg(data["methods"][key], "episode_difficulties")[:, -1]
        mean_d = float(diff.mean())
        std_d = float(diff.std())
        mae = float(np.abs(diff - target).mean())
        lines.append(f"{label} & {mean_d:.2f} & {std_d:.2f} & {mae:.2f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "difficulty_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    write_trajectory_summary(data)
    write_seed_table(data)
    write_difficulty_table(data)
    print("Wrote appendix tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())