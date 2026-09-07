"""Vectorized T1 layout and S2 target motion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

T1_XY = torch.tensor(((4.5, -1.0), (4.5, 1.0), (7.0, 3.5), (7.0, -3.5)))


@dataclass(frozen=True)
class T1S2CoreCfg:
    dt: float = 0.1
    target_surge_mps: float = 0.6
    target_lateral_amplitude_m: float = 3.0
    target_wavelength_m: float = 60.0
    ego_position_jitter_m: float = 0.35
    target_position_jitter_m: float = 0.20
    ego_yaw_jitter_rad: float = math.radians(12.0)


class T1S2Core:
    """Batched planar state used only by the fast policy-training task."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg=None, seed: int = 0) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg or T1S2CoreCfg()
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.ego_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.ego_yaw = torch.zeros(self.num_envs, device=self.device)
        self.ego_velocity = torch.zeros(self.num_envs, 2, device=self.device)
        self.ego_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self.ego_surge = torch.zeros(self.num_envs, device=self.device)
        self.target_base = T1_XY.to(self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.target_position = self.target_base.clone()
        self.target_velocity = torch.zeros_like(self.target_position)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.time = torch.zeros(self.num_envs, device=self.device)
        self.task_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.standoff = torch.full((self.num_envs,), 3.0, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        count = len(env_ids)
        rand = lambda *shape: torch.rand(*shape, generator=self.generator, device=self.device)
        self.ego_position[env_ids] = (2.0 * rand(count, 2) - 1.0) * self.cfg.ego_position_jitter_m
        self.ego_yaw[env_ids] = (2.0 * rand(count) - 1.0) * self.cfg.ego_yaw_jitter_rad
        self.ego_velocity[env_ids] = 0.0
        self.ego_yaw_rate[env_ids] = 0.0
        self.ego_surge[env_ids] = 0.0
        jitter = (2.0 * rand(count, 4, 2) - 1.0) * self.cfg.target_position_jitter_m
        self.target_base[env_ids] = T1_XY.to(self.device).unsqueeze(0) + jitter
        self.target_position[env_ids] = self.target_base[env_ids]
        self.target_velocity[env_ids] = 0.0
        self.phase[env_ids] = 2.0 * math.pi * rand(count)
        self.time[env_ids] = 0.0
        self.task_target[env_ids] = torch.randint(0, 2, (count,), generator=self.generator, device=self.device)
        distances = torch.tensor((2.0, 3.0, 4.0), device=self.device)
        self.standoff[env_ids] = distances[
            torch.randint(0, 3, (count,), generator=self.generator, device=self.device)
        ]
        if not torch.all(self.visible_target(env_ids)):
            raise RuntimeError("reset generated a task target outside the forward field of view")

    def advance_targets(self) -> None:
        self.time += self.cfg.dt
        omega = 2.0 * math.pi * self.cfg.target_surge_mps / self.cfg.target_wavelength_m
        angle = omega * self.time + self.phase
        lateral = -self.cfg.target_lateral_amplitude_m * (torch.sin(angle) - torch.sin(self.phase))
        lateral_velocity = -self.cfg.target_lateral_amplitude_m * omega * torch.cos(angle)
        self.target_position[..., 0] = self.target_base[..., 0] + self.cfg.target_surge_mps * self.time[:, None]
        self.target_position[..., 1] = self.target_base[..., 1]
        self.target_position[:, :2, 1] += lateral[:, None]
        self.target_velocity[..., 0] = self.cfg.target_surge_mps
        self.target_velocity[..., 1] = 0.0
        self.target_velocity[:, :2, 1] = lateral_velocity[:, None]

    def integrate_ego(self, world_velocity: torch.Tensor, yaw_rate: torch.Tensor) -> None:
        self.ego_velocity = world_velocity
        self.ego_yaw_rate = yaw_rate
        self.ego_position += world_velocity * self.cfg.dt
        self.ego_yaw = torch.atan2(
            torch.sin(self.ego_yaw + yaw_rate * self.cfg.dt),
            torch.cos(self.ego_yaw + yaw_rate * self.cfg.dt),
        )

    def relative_state(self):
        delta = self.target_position - self.ego_position[:, None, :]
        relative_velocity = self.target_velocity - self.ego_velocity[:, None, :]
        c, s = torch.cos(self.ego_yaw)[:, None], torch.sin(self.ego_yaw)[:, None]
        position = torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]), -1)
        velocity = torch.stack(
            (c * relative_velocity[..., 0] + s * relative_velocity[..., 1],
             -s * relative_velocity[..., 0] + c * relative_velocity[..., 1]), -1
        )
        velocity[..., 0] += self.ego_yaw_rate[:, None] * position[..., 1]
        velocity[..., 1] -= self.ego_yaw_rate[:, None] * position[..., 0]
        return position, velocity

    def task_position(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.target_position[env_ids, self.task_target]

    def task_range(self):
        return torch.linalg.norm(self.task_position() - self.ego_position, dim=-1)

    def visible_target(self, env_ids=None):
        position, _ = self.relative_state()
        index = torch.arange(self.num_envs, device=self.device)
        selected = position[index, self.task_target]
        visible = (selected[:, 0] > 1.0) & (torch.abs(torch.atan2(selected[:, 1], selected[:, 0])) < math.radians(55.0))
        return visible if env_ids is None else visible[env_ids]
