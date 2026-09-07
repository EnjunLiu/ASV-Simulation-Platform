"""Isaac Lab DirectRLEnv for semantic ASV policy fine-tuning."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv

from .appearance import AppearanceSampler
from .core import T1S2Core
from .kinematic_surrogate import KinematicSurrogate
from .observation import POLICY_OBS_DIM, NoisyEntityObservation


class SemanticAsvEnv(DirectRLEnv):
    """Four parallel T1/S2 environments with deployment-shaped observations."""

    cfg: "SemanticAsvEnvCfg"

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self.core = T1S2Core(self.num_envs, self.device, seed=int(self.cfg.seed))
        self.sensor = NoisyEntityObservation(self.num_envs, self.device, seed=int(self.cfg.seed) + 1)
        self.appearances = AppearanceSampler(self.cfg.appearance_library, self.device, seed=int(self.cfg.seed) + 2)
        self.surrogate = KinematicSurrogate(self.num_envs, self.device)
        self.task_table = self._load_task_table(self.cfg.task_embeddings)
        self.actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self._raw = torch.zeros(self.num_envs, 16, 516, device=self.device)
        self._mask = torch.zeros(self.num_envs, 16, dtype=torch.bool, device=self.device)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self) -> None:
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.8, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _load_task_table(self, path: str) -> torch.Tensor:
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"task embeddings not found: {path_obj}; see isaac/lab/README.md")
        with np.load(path_obj, allow_pickle=False) as blob:
            table = np.stack(
                [[blob[f"{color}_{distance}"] for distance in (2, 3, 4)] for color in ("red", "blue")]
            )
        return torch.as_tensor(table, dtype=torch.float32, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions = self.actions.clone()
        self.actions = actions.clone()
        norm = torch.linalg.norm(self.actions, dim=-1, keepdim=True)
        self.actions *= torch.clamp(0.5 / (norm + 1.0e-8), max=1.0)

    def _apply_action(self) -> None:
        world_velocity, yaw_rate, surge = self.surrogate.step(self.actions)
        self.core.ego_surge = surge
        self.core.integrate_ego(world_velocity, yaw_rate)
        self.core.advance_targets()
        relative_position, relative_velocity = self.core.relative_state()
        self._raw, self._mask = self.sensor.step(relative_position, relative_velocity)

    def _task_embedding(self) -> torch.Tensor:
        distance_index = self.core.standoff.to(torch.long) - 2
        return self.task_table[self.core.task_target, distance_index]

    def _get_observations(self):
        policy = torch.cat(
            (self._raw.reshape(self.num_envs, -1), self._mask.float(), self._task_embedding()), dim=-1
        )
        if policy.shape[-1] != POLICY_OBS_DIM:
            raise RuntimeError(f"unexpected policy observation shape {policy.shape}")
        return {"policy": policy}

    def _get_rewards(self) -> torch.Tensor:
        target_range = self.core.task_range()
        distance_error = torch.abs(target_range - self.core.standoff)
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_velocity = self.core.target_velocity[env_ids, self.core.task_target]
        velocity_error = torch.linalg.norm(target_velocity - self.core.ego_velocity, dim=-1)
        relative_position, _ = self.core.relative_state()
        selected = relative_position[env_ids, self.core.task_target]
        bearing = torch.abs(torch.atan2(selected[:, 1], selected[:, 0]))
        action_delta = torch.linalg.norm(self.actions - self.previous_actions, dim=-1)
        reward = (
            torch.exp(-distance_error / 0.75)
            + 0.25 * torch.exp(-velocity_error / 0.75)
            + 0.15 * torch.exp(-bearing / 0.6)
            - 0.05 * torch.square(action_delta)
            - 2.0 * torch.square(torch.relu(1.5 - target_range))
        )
        log = self.extras.setdefault("log", {})
        log["Metrics/mean_abs_standoff_error_m"] = distance_error.mean()
        log["Metrics/min_target_range_m"] = target_range.min()
        log["Metrics/mean_action_delta_m"] = action_delta.mean()
        return reward

    def _get_dones(self):
        target_range = self.core.task_range()
        relative_position, _ = self.core.relative_state()
        env_ids = torch.arange(self.num_envs, device=self.device)
        selected = relative_position[env_ids, self.core.task_target]
        bearing = torch.abs(torch.atan2(selected[:, 1], selected[:, 0]))
        terminated = (target_range < self.cfg.collision_range_m) | (target_range > self.cfg.lost_range_m)
        terminated |= bearing > self.cfg.lost_bearing_rad
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        self.core.reset(env_ids)
        self.surrogate.reset(env_ids, self.core.ego_yaw[env_ids])
        appearances = self.sensor._appearances.clone()
        appearances[env_ids] = self.appearances.sample(len(env_ids))
        relative_position, relative_velocity = self.core.relative_state()
        self.sensor.reset(env_ids, relative_position, relative_velocity, appearances)
        self._raw, self._mask = self.sensor.output()
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
