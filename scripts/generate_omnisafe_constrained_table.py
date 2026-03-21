from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "experiments" / "results" / "omnisafe_constrained_baseline_suite.json"
TABLE_PATH = ROOT / "paper" / "tables" / "constrained_baseline_comparison.tex"
MD_PATH = ROOT / "docs" / "OMNISAFE_CONSTRAINED_BASELINE_SUITE.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_p(entry: dict) -> str:
    sig = entry["significance"]
    if sig == "---":
        return "---"
    return f"{entry['p_value']:.3f} ({sig})"


def _permutation_p_value(a: list[float], b: list[float], n_shuffles: int = 20000, seed: int = 42) -> float:
    rng = random.Random(seed)
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pooled = list(a) + list(b)
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


def _sig_marker(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _row(env_label: str, algo_label: str, agg: dict, ptext: str, show_env: bool) -> str:
    env_cell = env_label if show_env else ""
    return (
        f"{env_cell} & {algo_label} & "
        f"{agg['final_ep_return']['mean']:.2f} $\\pm$ {agg['final_ep_return']['std']:.2f} & "
        f"{agg['final_ep_cost']['mean']:.3f} $\\pm$ {agg['final_ep_cost']['std']:.3f} & "
        f"{agg['final_ep_length']['mean']:.2f} $\\pm$ {agg['final_ep_length']['std']:.2f} & {ptext} \\\\" 
    )


def main() -> int:
    payload = _load_json(JSON_PATH)

    algo_cfg = payload.get("config", {}).get("algorithms", {})
    order = [(alias, name) for alias, name in algo_cfg.items()]
    if not order:
        raise ValueError("No algorithms found in artifact config.")
    envs = [("scaled", "Scaled"), ("defi", "DeFi")]

    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Env & Baseline & Final Return & Final Cost & Final Ep Len & $p$ vs best return \\",
        r"\midrule",
    ]

    for env_key, env_label in envs:
        returns_by_alias = {
            alias: [x["final_ep_return"] for x in payload["results"][env_key][alias]["per_seed"]]
            for alias, _ in order
        }
        best_alias = max(returns_by_alias, key=lambda k: statistics.mean(returns_by_alias[k]))

        for idx, (alias, algo_label) in enumerate(order):
            block = payload["results"][env_key][alias]
            if alias == best_alias:
                ptext = "---"
            else:
                p = _permutation_p_value(returns_by_alias[alias], returns_by_alias[best_alias])
                ptext = f"{p:.3f} ({_sig_marker(p)})"
            lines.append(
                _row(
                    env_label=env_label,
                    algo_label=algo_label,
                    agg=block["aggregate"],
                    ptext=ptext,
                    show_env=(idx == 0),
                )
            )
        if env_key != envs[-1][0]:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", "}"])

    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_lines = [
        "# OmniSafe Official Constrained Baselines",
        "",
        f"Generated (UTC): {payload.get('generated_at_utc', 'unknown')}",
        "",
        "Algorithms: " + ", ".join(name for _, name in order) + " (official OmniSafe implementations)",
        "",
    ]

    for env_key, env_label in envs:
        md_lines.append(f"## {env_label.upper()} Environment")
        md_lines.append("| Baseline | Final Return (mean±std) | Final Cost (mean±std) | Final Ep Len (mean±std) | p vs best return |")
        md_lines.append("|---|---:|---:|---:|---:|")

        returns_by_alias = {
            alias: [x["final_ep_return"] for x in payload["results"][env_key][alias]["per_seed"]]
            for alias, _ in order
        }
        best_alias = max(returns_by_alias, key=lambda k: statistics.mean(returns_by_alias[k]))

        for alias, algo_label in order:
            block = payload["results"][env_key][alias]
            agg = block["aggregate"]
            if alias == best_alias:
                ptext = "---"
            else:
                p = _permutation_p_value(returns_by_alias[alias], returns_by_alias[best_alias])
                ptext = f"{p:.3f} ({_sig_marker(p)})"

            md_lines.append(
                f"| {algo_label} | "
                f"{agg['final_ep_return']['mean']:.2f} ± {agg['final_ep_return']['std']:.2f} | "
                f"{agg['final_ep_cost']['mean']:.3f} ± {agg['final_ep_cost']['std']:.3f} | "
                f"{agg['final_ep_length']['mean']:.2f} ± {agg['final_ep_length']['std']:.2f} | "
                f"{ptext} |"
            )
        md_lines.append("")

    MD_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Source artifact: {JSON_PATH}")
    print(f"Generated at: {payload.get('generated_at_utc', 'unknown')}")
    print(f"Seed count: {len(payload.get('config', {}).get('seeds', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
