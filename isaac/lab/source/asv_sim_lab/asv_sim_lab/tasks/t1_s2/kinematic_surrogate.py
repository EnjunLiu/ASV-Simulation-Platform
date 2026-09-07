"""Fast differentiable-free response model for policy iteration throughput.

This module is intentionally not the ASV controller and not a hydrodynamic
model. It maps the desired body point to a plausible lagged planar response so
Isaac Lab can evaluate many policy rollouts quickly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KinematicSurrogateCfg:
    dt: float = 0.1
    max_action_m: float = 0.5
    max_surge_mps: float = 1.2
    max_yaw_rate_radps: float = 0.6
    surge_time_constant_s: float = 1.0
    yaw_time_constant_s: float = 0.35


class KinematicSurrogate:
    """Vectorized first-order surge/yaw response with no hidden dither."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg=None) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg or KinematicSurrogateCfg()
        self.surge = torch.zeros(self.num_envs, device=self.device)
        self.yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self.yaw = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor, yaw: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.surge[env_ids] = 0.0
        self.yaw_rate[env_ids] = 0.0
        self.yaw[env_ids] = yaw

    def step(self, actions: torch.Tensor):
        norm = torch.linalg.norm(actions, dim=-1)
        desired_surge = self.cfg.max_surge_mps * torch.clamp(norm / self.cfg.max_action_m, 0.0, 1.0)
        desired_heading_error = torch.atan2(actions[:, 1], actions[:, 0].clamp_min(1.0e-6))
        desired_yaw_rate = torch.clamp(
            desired_heading_error / self.cfg.yaw_time_constant_s,
            -self.cfg.max_yaw_rate_radps,
            self.cfg.max_yaw_rate_radps,
        )
        surge_alpha = min(1.0, self.cfg.dt / self.cfg.surge_time_constant_s)
        yaw_alpha = min(1.0, self.cfg.dt / self.cfg.yaw_time_constant_s)
        self.surge += surge_alpha * (desired_surge - self.surge)
        self.yaw_rate += yaw_alpha * (desired_yaw_rate - self.yaw_rate)
        self.yaw += self.yaw_rate * self.cfg.dt
        world_velocity = torch.stack(
            (self.surge * torch.cos(self.yaw), self.surge * torch.sin(self.yaw)), dim=-1
        )
        return world_velocity, self.yaw_rate, self.surge
