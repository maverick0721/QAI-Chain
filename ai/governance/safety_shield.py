from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


class ShieldDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    CLAMP = "CLAMP"


@dataclass
class ShieldState:
    difficulty: float
    step: int
    state_vector: list[float]


class AuditSink(Protocol):
    def commit_audit_record(self, record: dict[str, object]) -> str:
        ...


class GovernanceSafetyShield:
    MIN_DIFFICULTY = 1
    MAX_DIFFICULTY = 15
    MAX_STEP_SIZE = 2.0
    MIN_INTERVAL = 3

    def __init__(self, audit_sink: AuditSink | None = None, history_size: int = 512):
        self.audit_sink = audit_sink
        self._last_adjustment_step = -10**9
        self._action_history: deque[float] = deque(maxlen=history_size)

    def _is_anomalous(self, action: float) -> bool:
        if len(self._action_history) < 25:
            return False
        mu = float(np.mean(self._action_history))
        sigma = float(np.std(self._action_history) + 1e-8)
        return abs(action - mu) > 3.0 * sigma

    def validate_action(self, proposed_action: float, current_state: ShieldState) -> tuple[ShieldDecision, float]:
        action = float(proposed_action)
        if current_state.step - self._last_adjustment_step < self.MIN_INTERVAL and abs(action) > 0.0:
            action = 0.0
            decision = ShieldDecision.REJECT
            self._log(current_state, proposed_action, action, decision, "min_interval")
            return decision, action

        new_difficulty = current_state.difficulty + action
        if new_difficulty < self.MIN_DIFFICULTY or new_difficulty > self.MAX_DIFFICULTY:
            decision = ShieldDecision.REJECT
            self._log(current_state, proposed_action, 0.0, decision, "hard_bounds")
            return decision, 0.0

        if abs(action) > self.MAX_STEP_SIZE:
            action = float(np.clip(action, -self.MAX_STEP_SIZE, self.MAX_STEP_SIZE))
            decision = ShieldDecision.CLAMP
            self._accept_bookkeeping(action, current_state.step)
            self._log(current_state, proposed_action, action, decision, "step_clamp")
            return decision, action

        if self._is_anomalous(action):
            decision = ShieldDecision.REJECT
            self._log(current_state, proposed_action, 0.0, decision, "anomaly_3sigma")
            return decision, 0.0

        decision = ShieldDecision.ACCEPT
        self._accept_bookkeeping(action, current_state.step)
        self._log(current_state, proposed_action, action, decision, "ok")
        return decision, action

    def _accept_bookkeeping(self, action: float, step: int) -> None:
        self._action_history.append(float(action))
        if abs(action) > 0.0:
            self._last_adjustment_step = step

    def _log(
        self,
        current_state: ShieldState,
        proposed_action: float,
        executed_action: float,
        decision: ShieldDecision,
        reason: str,
    ) -> None:
        if self.audit_sink is None:
            return
        self.audit_sink.commit_audit_record(
            {
                "event": "shield_validation",
                "step": int(current_state.step),
                "state": list(current_state.state_vector),
                "difficulty": float(current_state.difficulty),
                "proposed_action": float(proposed_action),
                "executed_action": float(executed_action),
                "decision": decision.value,
                "reason": reason,
            }
        )
