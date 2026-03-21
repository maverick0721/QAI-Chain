from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ScaledGovernanceConfig:
    episode_length: int = 200
    state_dim: int = 12
    action_dim: int = 3
    noise_std: float = 0.1
    min_difficulty: float = 1.0
    max_difficulty: float = 15.0
    trace_csv_path: str | None = None


class ScaledGovernanceEnv:
    """12D governance state with adversarial scenarios and partial observability."""

    def __init__(self, cfg: ScaledGovernanceConfig | None = None):
        self.cfg = cfg or ScaledGovernanceConfig()
        self.step_idx = 0
        self._eclipse_left = 0
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
        self.step_idx = 0
        self.difficulty = 7.0
        self.block_size = 1.0
        self.gas_limit = 1.0
        self.peer_count = 120.0
        self.fork_depth = 0.0
        self.mempool_diversity = 0.7
        self.partition_score = 0.1
        self.pqc_load = 0.15
        self.quantum_latency = 0.25
        self.attack_pressure = 0.2
        self.delay = 1.0
        self.burst = 0.0
        self.telemetry_quality = 1.0
        self._eclipse_left = 0
        return self.observe()

    def _true_state(self) -> np.ndarray:
        return np.array(
            [
                self.difficulty,
                self.block_size,
                self.gas_limit,
                self.peer_count,
                self.fork_depth,
                self.mempool_diversity,
                self.partition_score,
                self.pqc_load,
                self.quantum_latency,
                self.attack_pressure,
                self.delay,
                self.burst,
            ],
            dtype=np.float32,
        )

    def observe(self) -> np.ndarray:
        s = self._true_state().copy()
        noisy_idx = [3, 4, 6, 10]
        s[noisy_idx] += np.random.normal(0.0, self.cfg.noise_std, size=len(noisy_idx))
        if self._eclipse_left > 0:
            s[3] *= 0.6
            s[6] = min(1.0, s[6] + 0.25)
            self._eclipse_left -= 1
        return s.astype(np.float32)

    def _apply_adversary(self, intensity: float) -> None:
        trace_idx = max(0, self.step_idx - 1)
        intensity = float(np.clip(self._trace_value(trace_idx, "intensity", intensity), 0.0, 1.5))
        p_sybil = 0.30 * intensity
        p_eclipse = 0.08 * intensity
        p_mev = 0.22 * intensity
        p_cascade = 0.10 * intensity

        if np.random.rand() < p_sybil:
            self.telemetry_quality = 0.7
            self.partition_score = min(1.0, self.partition_score + 0.08)
        else:
            self.telemetry_quality = 1.0

        if np.random.rand() < p_eclipse:
            self._eclipse_left = 10

        if np.random.rand() < p_mev:
            self.mempool_diversity = max(0.0, self.mempool_diversity - 0.10)
            self.burst += 0.5

        if np.random.rand() < p_cascade:
            self.attack_pressure = min(2.0, self.attack_pressure + 0.35)
            self.delay = min(12.0, self.delay + 0.8)
            self.burst = min(4.0, self.burst + 0.9)

    def step(self, action: np.ndarray | list[float], adversarial_intensity: float = 0.5):
        self.step_idx += 1
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.cfg.action_dim:
            raise ValueError(f"Expected action_dim={self.cfg.action_dim}, got {a.shape[0]}")

        self._apply_adversary(float(np.clip(adversarial_intensity, 0.0, 1.0)))

        self.difficulty = float(np.clip(self.difficulty + 1.0 * a[0], self.cfg.min_difficulty, self.cfg.max_difficulty))
        self.block_size = float(np.clip(self.block_size + 0.2 * a[1], 0.4, 3.0))
        self.gas_limit = float(np.clip(self.gas_limit + 0.2 * a[2], 0.4, 3.0))

        trace_idx = max(0, self.step_idx - 1)
        demand_base = self._trace_value(trace_idx, "demand_base", 8.0)
        demand_std = max(0.05, self._trace_value(trace_idx, "demand_std", 1.2))
        demand = max(0.0, np.random.normal(demand_base + self.attack_pressure + self.burst, demand_std))
        service = max(0.1, (self.block_size * self.gas_limit) * (10.0 / (2.5 + self.difficulty)))
        service = service / (1.0 + 0.5 * self.attack_pressure)

        backlog = max(0.0, demand - service)
        self.delay = float(max(0.2, 0.92 * self.delay + 0.18 * backlog + np.random.normal(0.0, 0.08)))
        self.fork_depth = float(max(0.0, 0.55 * self.fork_depth + 0.05 * self.attack_pressure + 0.03 * max(0.0, self.delay - 1.0)))
        self.partition_score = float(np.clip(0.9 * self.partition_score + 0.03 * self.attack_pressure + np.random.normal(0.0, 0.01), 0.0, 1.0))
        self.pqc_load = float(np.clip(0.9 * self.pqc_load + 0.03 * self.gas_limit + 0.01 * self.attack_pressure, 0.0, 1.0))
        self.quantum_latency = float(np.clip(0.92 * self.quantum_latency + 0.04 * self.attack_pressure + np.random.normal(0.0, 0.01), 0.05, 2.0))
        attack_drift = self._trace_value(trace_idx, "attack_drift", 0.18)
        attack_noise = max(0.01, self._trace_value(trace_idx, "attack_noise", 0.08))
        self.attack_pressure = float(np.clip(0.87 * self.attack_pressure + np.random.normal(attack_drift, attack_noise), 0.0, 2.0))
        self.burst = float(max(0.0, 0.7 * self.burst + np.random.normal(0.0, 0.12)))
        self.peer_count = float(max(20.0, self.peer_count + np.random.normal(0.0, 2.0) - 2.5 * self.partition_score))
        self.mempool_diversity = float(np.clip(self.mempool_diversity + np.random.normal(0.0, 0.02) - 0.04 * self.attack_pressure, 0.0, 1.0))

        target_difficulty = 4.0
        target_error = abs(self.difficulty - target_difficulty)
        safety_violation = float(
            self.delay > 7.0
            or self.fork_depth > 1.2
            or self.partition_score > 0.85
            or self.difficulty in (self.cfg.min_difficulty, self.cfg.max_difficulty)
        )

        reward = (
            1.25 * service
            - 1.05 * self.delay
            - 1.10 * self.attack_pressure
            - 1.20 * self.fork_depth
            - 0.45 * target_error
            - 2.0 * safety_violation
        )

        done = self.step_idx >= self.cfg.episode_length
        info = {
            "target_error": float(target_error),
            "safety_violation": float(safety_violation),
            "delay": float(self.delay),
            "fork_depth": float(self.fork_depth),
            "partition_score": float(self.partition_score),
        }
        return self.observe(), float(reward), bool(done), info
