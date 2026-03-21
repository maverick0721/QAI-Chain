from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DeFiGovConfig:
    episode_length: int = 150
    state_dim: int = 8
    action_dim: int = 2  # fee adjustment, rebalance threshold adjustment
    trace_csv_path: str | None = None


class DeFiLiquidityGovernanceEnv:
    """Simplified DeFi pool governance with MEV-style adversarial pressure."""

    def __init__(self, cfg: DeFiGovConfig | None = None):
        self.cfg = cfg or DeFiGovConfig()
        self._trace = self._load_trace(self.cfg.trace_csv_path)
        self.reset()

    @staticmethod
    def _load_trace(trace_csv_path: str | None) -> pd.DataFrame | None:
        if not trace_csv_path:
            return None
        trace_path = Path(trace_csv_path)
        if not trace_path.exists():
            return None
        trace_df = pd.read_csv(trace_path)
        return trace_df if not trace_df.empty else None

    def _trace_value(self, idx: int, col: str, default: float) -> float:
        if self._trace is None or col not in self._trace.columns:
            return default
        row = self._trace.iloc[idx % len(self._trace)]
        val = row[col]
        if pd.isna(val):
            return default
        return float(val)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.fee = 0.003
        self.rebalance_threshold = 0.15
        self.depth = 1.0
        self.volatility = 0.2
        self.mev_pressure = 0.15
        self.slippage = 0.02
        self.lp_churn = 0.1
        self.oracle_noise = 0.03
        return self._state()

    def _state(self) -> np.ndarray:
        return np.array(
            [
                self.fee,
                self.rebalance_threshold,
                self.depth,
                self.volatility,
                self.mev_pressure,
                self.slippage,
                self.lp_churn,
                self.oracle_noise,
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray | list[float], adversarial_intensity: float = 0.5):
        self.t += 1
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.cfg.action_dim:
            raise ValueError(f"Expected action_dim={self.cfg.action_dim}, got {a.shape[0]}")

        trace_idx = max(0, self.t - 1)
        adversarial_intensity = float(np.clip(self._trace_value(trace_idx, "intensity", adversarial_intensity), 0.0, 1.5))

        self.fee = float(np.clip(self.fee + 0.0008 * a[0], 0.0005, 0.02))
        self.rebalance_threshold = float(np.clip(self.rebalance_threshold + 0.02 * a[1], 0.03, 0.5))

        vol_drift = self._trace_value(trace_idx, "volatility_drift", 0.02)
        vol_noise = max(0.005, self._trace_value(trace_idx, "volatility_noise", 0.03))
        mev_drift = self._trace_value(trace_idx, "mev_drift", 0.25)
        mev_noise = max(0.005, self._trace_value(trace_idx, "mev_noise", 0.04))
        oracle_drift = self._trace_value(trace_idx, "oracle_drift", 0.05)
        oracle_noise = max(0.001, self._trace_value(trace_idx, "oracle_noise", 0.01))

        self.volatility = float(np.clip(0.90 * self.volatility + np.random.normal(vol_drift, vol_noise), 0.01, 1.2))
        self.mev_pressure = float(np.clip(0.88 * self.mev_pressure + mev_drift * adversarial_intensity + np.random.normal(0.0, mev_noise), 0.0, 2.0))
        self.oracle_noise = float(np.clip(0.9 * self.oracle_noise + oracle_drift * adversarial_intensity + np.random.normal(0.0, oracle_noise), 0.0, 0.5))

        flow_std = max(0.05, self._trace_value(trace_idx, "swap_flow_std", 0.3))
        swap_flow = max(0.0, np.random.normal(1.0 + self.volatility, flow_std))
        mev_loss = max(0.0, self.mev_pressure * (0.8 - 30.0 * self.fee))
        rebalance_gain = max(0.0, (0.5 - self.rebalance_threshold) * self.depth)

        self.slippage = float(np.clip(0.75 * self.slippage + 0.02 * swap_flow + 0.05 * self.mev_pressure, 0.001, 0.6))
        self.lp_churn = float(np.clip(0.8 * self.lp_churn + 0.03 * self.slippage + 0.01 * self.oracle_noise, 0.0, 1.0))
        self.depth = float(max(0.1, self.depth + 0.15 * (0.0025 - self.fee) - 0.2 * self.lp_churn + 0.1 * rebalance_gain))

        pnl = 1.7 * self.fee * swap_flow + 0.8 * rebalance_gain - 1.2 * self.slippage - 1.1 * mev_loss - 0.7 * self.lp_churn
        safety_violation = float(self.slippage > 0.35 or self.depth < 0.2)
        reward = float(pnl - 1.5 * safety_violation)

        done = self.t >= self.cfg.episode_length
        info = {
            "safety_violation": safety_violation,
            "slippage": float(self.slippage),
            "depth": float(self.depth),
            "mev_pressure": float(self.mev_pressure),
        }
        return self._state(), reward, bool(done), info
