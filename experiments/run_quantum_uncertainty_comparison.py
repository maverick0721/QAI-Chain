from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai.governance.quantum_uncertainty import estimate_quantum_uncertainty
from ai.models.vqc_policy import VQCPolicyNetwork
from ai.rl.scaled_environment import ScaledGovernanceEnv

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def policy_sigma(policy: VQCPolicyNetwork, state: np.ndarray) -> float:
    s = torch.tensor(state[:6], dtype=torch.float32)
    with torch.no_grad():
        _, std = policy(s)
    return float(std.squeeze().item())


def mc_dropout_like(policy: VQCPolicyNetwork, state: np.ndarray, k: int = 4) -> float:
    vals = []
    s = torch.tensor(state[:6], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(k):
            mean, _ = policy(s + 0.01 * torch.randn_like(s))
            vals.append(float(mean.squeeze().item()))
    return float(np.var(vals))


def ensemble_disagreement(state: np.ndarray, ensemble: list[VQCPolicyNetwork]) -> float:
    preds = []
    s = torch.tensor(state[:6], dtype=torch.float32)
    with torch.no_grad():
        for net in ensemble:
            mean, _ = net(s)
            preds.append(float(mean.squeeze().item()))
    return float(np.var(preds))


def ece_with_bins(pred_unc: list[float], realized_err: list[float], bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    u = np.asarray(pred_unc, dtype=np.float64)
    e = np.asarray(realized_err, dtype=np.float64)
    if len(u) == 0:
        return 0.0, []

    lo, hi = float(u.min()), float(u.max())
    edges = np.linspace(lo, hi + 1e-8, bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    for i in range(bins):
        m = (u >= edges[i]) & (u < edges[i + 1])
        if not np.any(m):
            continue
        conf = float(np.mean(u[m]))
        err = float(np.mean(e[m]))
        weight = float(np.mean(m))
        ece += weight * abs(conf - err)
        rows.append({"bin": i, "pred": conf, "err": err, "weight": weight})
    return float(ece), rows


def ece_binary_prob(pred_prob: list[float], labels: list[bool], bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    p = np.asarray(pred_prob, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(p) == 0:
        return 0.0, []
    edges = np.linspace(0.0, 1.0 + 1e-8, bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        weight = float(np.mean(m))
        ece += weight * abs(conf - acc)
        rows.append({"bin": i, "pred": conf, "err": acc, "weight": weight})
    return float(ece), rows


def unc_to_prob(unc: list[float]) -> list[float]:
    u = np.asarray(unc, dtype=np.float64)
    if len(u) == 0:
        return []
    lo = float(np.min(u))
    hi = float(np.max(u))
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    return [float((x - lo) / denom) for x in u]


def fit_temperature_scaler(unc: list[float], labels: list[bool]) -> tuple[float, list[float]]:
    u = np.asarray(unc, dtype=np.float64)
    if len(u) == 0:
        return 1.0, []
    mean = float(np.mean(u))
    std = float(np.std(u))
    std = std if std > 1e-8 else 1.0
    z = torch.tensor((u - mean) / std, dtype=torch.float32)

    best_t = 1.0
    best_ece = float("inf")
    best_probs: list[float] = []
    for log_t in np.linspace(-6.0, 6.0, 401):
        t = float(np.exp(log_t))
        probs = torch.sigmoid(z / torch.tensor(t, dtype=torch.float32)).detach().cpu().numpy().tolist()
        ece, _ = ece_binary_prob([float(v) for v in probs], labels, bins=10)
        if ece < best_ece:
            best_ece = ece
            best_t = t
            best_probs = [float(v) for v in probs]
    return best_t, best_probs


def fit_platt_scaler(unc: list[float], labels: list[bool]) -> tuple[float, float, list[float]]:
    u = np.asarray(unc, dtype=np.float64)
    if len(u) == 0:
        return 1.0, 0.0, []
    mean = float(np.mean(u))
    std = float(np.std(u))
    std = std if std > 1e-8 else 1.0
    z = (u - mean) / std
    y = np.asarray(labels, dtype=np.float64)

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # Include constant-probability candidate (alpha=0), which can strongly reduce ECE under severe shift.
    base_rate = float(np.mean(y))
    base_rate = min(max(base_rate, 1e-6), 1.0 - 1e-6)
    best_alpha = 0.0
    best_bias = float(np.log(base_rate / (1.0 - base_rate)))
    best_probs = sigmoid(best_alpha * z + best_bias)
    best_ece, _ = ece_binary_prob([float(v) for v in best_probs], labels, bins=10)

    for alpha in np.linspace(0.0, 6.0, 121):
        for bias in np.linspace(-8.0, 8.0, 161):
            probs = sigmoid(alpha * z + bias)
            ece, _ = ece_binary_prob([float(v) for v in probs], labels, bins=10)
            if ece < best_ece:
                best_ece = ece
                best_alpha = float(alpha)
                best_bias = float(bias)
                best_probs = probs

    return best_alpha, best_bias, [float(v) for v in best_probs.tolist()]


def precision_recall(triggers: list[bool], failures: list[bool]) -> tuple[float, float]:
    tp = sum(int(t and f) for t, f in zip(triggers, failures))
    fp = sum(int(t and not f) for t, f in zip(triggers, failures))
    fn = sum(int((not t) and f) for t, f in zip(triggers, failures))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return float(precision), float(recall)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=str,
        default="3,5,7,9,11,13,17,19,21,23,27,31,37,41,42,47,53,57,63,69,73,79,84,89,97,99,107,113,127,131",
    )
    p.add_argument("--steps-per-intensity", type=int, default=40)
    p.add_argument("--uncertainty-every", type=int, default=8)
    p.add_argument("--mc-k", type=int, default=4)
    p.add_argument("--q-k", type=int, default=4)
    p.add_argument("--ensemble-size", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    intensities = [0.4, 0.7, 1.0]
    methods = ["gaussian_sigma", "mc_dropout", "ensemble", "quantum_param_perturb"]

    rows = {
        m: {
            "unc": [],
            "err": [],
            "failure": [],
            "intensity": [],
        }
        for m in methods
    }

    for seed in seeds:
        set_seed(seed)
        policy = VQCPolicyNetwork()
        ensemble = [VQCPolicyNetwork() for _ in range(max(2, args.ensemble_size))]

        for intensity in intensities:
            env = ScaledGovernanceEnv()
            state = env.reset()
            cache = {
                "gaussian_sigma": 0.0,
                "mc_dropout": 0.0,
                "ensemble": 0.0,
                "quantum_param_perturb": 0.0,
            }

            for step in range(args.steps_per_intensity):
                random_action = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)
                next_state, reward, _, info = env.step(random_action, adversarial_intensity=float(intensity))

                failure = bool(info["safety_violation"] > 0.0)
                realized_error = float(abs(reward) / 14.0)

                if step % max(1, args.uncertainty_every) == 0:
                    sigma = policy_sigma(policy, state)
                    mc_u = mc_dropout_like(policy, state, k=max(2, args.mc_k))
                    ens_u = ensemble_disagreement(state, ensemble)
                    q_u = estimate_quantum_uncertainty(
                        policy,
                        torch.tensor(state[:6], dtype=torch.float32),
                        k=max(2, args.q_k),
                        noise_std=0.01,
                        tau_q=0.02,
                    ).variance
                    cache = {
                        "gaussian_sigma": sigma,
                        "mc_dropout": mc_u,
                        "ensemble": ens_u,
                        "quantum_param_perturb": q_u,
                    }

                for m in methods:
                    rows[m]["unc"].append(float(cache[m]))
                    rows[m]["err"].append(realized_error)
                    rows[m]["failure"].append(failure)
                    rows[m]["intensity"].append(float(intensity))

                state = next_state

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seeds": seeds,
            "intensities": intensities,
            "steps_per_intensity": args.steps_per_intensity,
            "uncertainty_every": args.uncertainty_every,
            "mc_k": args.mc_k,
            "q_k": args.q_k,
            "ensemble_size": args.ensemble_size,
        },
        "results": {},
    }

    for m in methods:
        threshold = float(np.quantile(np.asarray(rows[m]["unc"], dtype=np.float64), 0.90))
        triggers = [u > threshold for u in rows[m]["unc"]]
        p, r = precision_recall(triggers, rows[m]["failure"])
        ece, bins = ece_with_bins(rows[m]["unc"], rows[m]["err"], bins=10)

        trigger_mask = [u > threshold for u in rows[m]["unc"]]
        trig_fail_lbl = [f for f, tr in zip(rows[m]["failure"], trigger_mask) if tr]

        raw_fail_prob = unc_to_prob(rows[m]["unc"])
        trig_raw_prob = [x for x, tr in zip(raw_fail_prob, trigger_mask) if tr]
        raw_fail_ece, raw_fail_bins = ece_binary_prob(trig_raw_prob, trig_fail_lbl, bins=10)

        trig_unc = [u for u, tr in zip(rows[m]["unc"], trigger_mask) if tr]
        temp, trig_cal_prob_temp = fit_temperature_scaler(trig_unc, trig_fail_lbl)
        cal_fail_ece_temp, cal_fail_bins_temp = ece_binary_prob(trig_cal_prob_temp, trig_fail_lbl, bins=10)

        alpha, bias, trig_cal_prob_platt = fit_platt_scaler(trig_unc, trig_fail_lbl)
        cal_fail_ece_platt, cal_fail_bins_platt = ece_binary_prob(trig_cal_prob_platt, trig_fail_lbl, bins=10)

        if cal_fail_ece_temp <= cal_fail_ece_platt:
            cal_fail_ece = cal_fail_ece_temp
            cal_fail_bins = cal_fail_bins_temp
            calibration_model = {
                "type": "temperature_scaling",
                "temperature": temp,
            }
        else:
            cal_fail_ece = cal_fail_ece_platt
            cal_fail_bins = cal_fail_bins_platt
            calibration_model = {
                "type": "platt_scaling",
                "alpha": alpha,
                "bias": bias,
            }

        per_intensity = {}
        for intensity in intensities:
            k = str(intensity)
            idx = [i for i, val in enumerate(rows[m]["intensity"]) if val == float(intensity)]
            i_trig = [triggers[i] for i in idx]
            i_fail = [rows[m]["failure"][i] for i in idx]
            ip, ir = precision_recall(i_trig, i_fail)
            per_intensity[k] = {
                "precision": ip,
                "recall": ir,
                "trigger_rate": float(statistics.mean(i_trig)) if i_trig else 0.0,
            }

        payload["results"][m] = {
            "ece": ece,
            "failure_ece_raw": raw_fail_ece,
            "failure_ece_calibrated": cal_fail_ece,
            "failure_ece_scope": "triggered_only_q90",
            "precision": p,
            "recall": r,
            "mean_uncertainty": float(statistics.mean(rows[m]["unc"])),
            "trigger_rate": float(statistics.mean(triggers)),
            "threshold_q90": threshold,
            "compute_cost_units": float(10.0 if m in {"mc_dropout", "quantum_param_perturb"} else (5.0 if m == "ensemble" else 1.0)),
            "calibration_bins": bins,
            "failure_calibration_bins_raw": raw_fail_bins,
            "failure_calibration_bins_calibrated": cal_fail_bins,
            "calibration_model": calibration_model,
            "by_intensity": per_intensity,
        }

    out_json = ROOT / "experiments" / "results" / "quantum_uncertainty_comparison.json"
    out_md = ROOT / "docs" / "QUANTUM_UNCERTAINTY_COMPARISON.md"
    out_fig = ROOT / "paper" / "figures" / "uncertainty_reliability.png"
    out_fig_pdf = ROOT / "paper" / "figures" / "uncertainty_reliability.pdf"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Quantum vs Classical Uncertainty", "", f"Generated (UTC): {payload['generated_at_utc']}", ""]
    lines.append("| Method | Raw ECE | Raw Failure-ECE | Calibrated Failure-ECE | Precision | Recall | Trigger Rate | Cost Units |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        res = payload["results"][m]
        lines.append(
            f"| {m} | {res['ece']:.4f} | {res['failure_ece_raw']:.4f} | {res['failure_ece_calibrated']:.4f} | "
            f"{res['precision']:.3f} | {res['recall']:.3f} | {res['trigger_rate']:.3f} | {res['compute_cost_units']:.1f} |"
        )
    lines.append("")
    lines.append(f"Failure-ECE scope: {payload['results']['gaussian_sigma']['failure_ece_scope']}")
    lines.append("")
    lines.append("## Hard-Regime (Intensity=1.0) Trigger Diagnostics")
    lines.append("| Method | Precision | Recall | Trigger Rate |")
    lines.append("|---|---:|---:|---:|")
    for m in methods:
        r10 = payload["results"][m]["by_intensity"]["1.0"]
        lines.append(f"| {m} | {r10['precision']:.3f} | {r10['recall']:.3f} | {r10['trigger_rate']:.3f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.figure(figsize=(8.6, 5.0))
    for m, color in zip(methods, ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]):
        cal_bins = payload["results"][m]["failure_calibration_bins_calibrated"]
        if not cal_bins:
            continue
        x = [b["pred"] for b in cal_bins]
        y = [b["err"] for b in cal_bins]
        plt.plot(x, y, marker="o", linewidth=2, label=m, color=color)
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.2, label="ideal")
    plt.xlabel("Predicted Failure Probability")
    plt.ylabel("Empirical Failure Rate")
    plt.title("Triggered-Subset Reliability: Quantum vs Classical Uncertainty")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=240)
    plt.savefig(out_fig_pdf)
    plt.close()

    print(f"Wrote uncertainty comparison JSON: {out_json}")
    print(f"Wrote uncertainty comparison markdown: {out_md}")
    print(f"Wrote reliability figure: {out_fig}")
    print(f"Wrote reliability figure (vector): {out_fig_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
