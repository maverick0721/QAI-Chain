from __future__ import annotations

from dataclasses import dataclass

from core.utils.utils import current_timestamp, serialize, sha256


@dataclass
class AuditRecord:
    epoch: int
    policy_hash: str
    state: list[float]
    action: list[float] | float
    uncertainty: float
    fallback_triggered: bool
    shield_result: str
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "policy_hash": self.policy_hash,
            "state": self.state,
            "action": self.action,
            "uncertainty": self.uncertainty,
            "fallback_triggered": self.fallback_triggered,
            "shield_result": self.shield_result,
            "timestamp": self.timestamp,
        }


def hash_audit_record(record: dict[str, object]) -> str:
    return sha256(serialize(record))


def make_audit_record(
    epoch: int,
    policy_hash: str,
    state: list[float],
    action: list[float] | float,
    uncertainty: float,
    fallback_triggered: bool,
    shield_result: str,
) -> dict[str, object]:
    rec = AuditRecord(
        epoch=epoch,
        policy_hash=policy_hash,
        state=state,
        action=action,
        uncertainty=uncertainty,
        fallback_triggered=fallback_triggered,
        shield_result=shield_result,
        timestamp=current_timestamp(),
    )
    payload = rec.to_dict()
    payload["record_hash"] = hash_audit_record(payload)
    return payload
