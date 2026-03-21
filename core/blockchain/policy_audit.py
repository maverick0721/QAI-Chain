from __future__ import annotations

import hashlib

import torch


def hash_policy_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous().view(-1)
        h.update(key.encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def verify_epoch_record(epoch_record: dict[str, object], policy_state_dict: dict[str, torch.Tensor]) -> bool:
    return hash_policy_state_dict(policy_state_dict) == str(epoch_record.get("policy_hash", ""))
