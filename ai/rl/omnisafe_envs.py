from __future__ import annotations

import os
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register

from ai.rl.defi_environment import DeFiGovConfig, DeFiLiquidityGovernanceEnv
from ai.rl.scaled_environment import ScaledGovernanceConfig, ScaledGovernanceEnv


class _BaseQAIChainCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = []
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id)
        self._num_envs = 1
        self._metadata = {}
        self._episode_steps = int(kwargs.get("episode_steps", self._episode_steps_from_env()))
        self._adversarial_intensity = float(kwargs.get("adversarial_intensity", self._intensity_from_env()))
        self._max_episode_steps = self._episode_steps
        self._t = 0
        self._env = self.build_env(self._episode_steps)

    @classmethod
    def default_episode_steps(cls) -> int:
        raise NotImplementedError

    @classmethod
    def episode_steps_env_var(cls) -> str:
        raise NotImplementedError

    def _episode_steps_from_env(self) -> int:
        raw = os.getenv(self.episode_steps_env_var())
        return int(raw) if raw else self.default_episode_steps()

    def _intensity_from_env(self) -> float:
        raw = os.getenv("QAI_OMNISAFE_ADVERSARIAL_INTENSITY")
        return float(raw) if raw else 0.7

    def build_env(self, episode_steps: int):
        raise NotImplementedError

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self.set_seed(seed)
        self._t = 0
        obs = self._env.reset()
        return torch.as_tensor(obs, dtype=torch.float32), {}

    def render(self) -> Any:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)


@env_register
class QAIChainScaledCMDP(_BaseQAIChainCMDP):
    _support_envs: ClassVar[list[str]] = ["QAIChainScaled-v0"]

    @classmethod
    def default_episode_steps(cls) -> int:
        return 140

    @classmethod
    def episode_steps_env_var(cls) -> str:
        return "QAI_OMNISAFE_EPISODE_STEPS_SCALED"

    def build_env(self, episode_steps: int) -> ScaledGovernanceEnv:
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        return ScaledGovernanceEnv(ScaledGovernanceConfig(episode_length=episode_steps))

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._t += 1
        act = torch.clamp(action, -1.0, 1.0).detach().cpu().numpy()
        obs, reward, done, info = self._env.step(act, adversarial_intensity=self._adversarial_intensity)
        cost = float(info.get("safety_violation", 0.0))
        terminated = bool(done)
        truncated = bool(self._t >= self._max_episode_steps)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        info_out: dict[str, Any] = {
            "target_error": float(info.get("target_error", 0.0)),
            "safety_violation": float(cost),
            "final_observation": obs_t,
        }
        return (
            obs_t,
            torch.as_tensor(float(reward), dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.float32),
            torch.as_tensor(truncated, dtype=torch.float32),
            info_out,
        )


@env_register
class QAIChainDeFiCMDP(_BaseQAIChainCMDP):
    _support_envs: ClassVar[list[str]] = ["QAIChainDeFi-v0"]

    @classmethod
    def default_episode_steps(cls) -> int:
        return 120

    @classmethod
    def episode_steps_env_var(cls) -> str:
        return "QAI_OMNISAFE_EPISODE_STEPS_DEFI"

    def build_env(self, episode_steps: int) -> DeFiLiquidityGovernanceEnv:
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        return DeFiLiquidityGovernanceEnv(DeFiGovConfig(episode_length=episode_steps))

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._t += 1
        act = torch.clamp(action, -1.0, 1.0).detach().cpu().numpy()
        obs, reward, done, info = self._env.step(act, adversarial_intensity=self._adversarial_intensity)
        cost = float(info.get("safety_violation", 0.0))
        terminated = bool(done)
        truncated = bool(self._t >= self._max_episode_steps)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        info_out: dict[str, Any] = {
            "slippage": float(info.get("slippage", 0.0)),
            "depth": float(info.get("depth", 0.0)),
            "safety_violation": float(cost),
            "final_observation": obs_t,
        }
        return (
            obs_t,
            torch.as_tensor(float(reward), dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.float32),
            torch.as_tensor(truncated, dtype=torch.float32),
            info_out,
        )