"""Deployment-shaped noisy entity observations for parallel PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

E_MAX = 16
ENTITY_COUNT = 4
KIN_DIM = 4
OWL_DIM = 512
RAW_DIM = KIN_DIM + OWL_DIM
TASK_DIM = 64
POLICY_OBS_DIM = E_MAX * RAW_DIM + E_MAX + TASK_DIM


@dataclass(frozen=True)
class EntityObservationCfg:
    control_dt: float = 0.1
    detection_period_steps: int = 5
    delay_steps: int = 1
    position_noise_std_m: float = 0.04
    velocity_noise_std_mps: float = 0.03
    dropout_probability: float = 0.05
    hold_detection_misses: int = 6


class NoisyEntityObservation:
    """Emulate delayed 2 Hz detections and 10 Hz tracker prediction."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg=None, seed: int = 0) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg or EntityObservationCfg()
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        history = max(1, int(self.cfg.delay_steps) + 1)
        self._position_history = torch.zeros(history, self.num_envs, ENTITY_COUNT, 2, device=self.device)
        self._velocity_history = torch.zeros_like(self._position_history)
        self._history_index = 0
        self._tracked_position = torch.zeros(self.num_envs, ENTITY_COUNT, 2, device=self.device)
        self._tracked_velocity = torch.zeros_like(self._tracked_position)
        self._appearances = torch.zeros(self.num_envs, ENTITY_COUNT, OWL_DIM, device=self.device)
        self._valid = torch.zeros(self.num_envs, ENTITY_COUNT, dtype=torch.bool, device=self.device)
        self._misses = torch.zeros(self.num_envs, ENTITY_COUNT, dtype=torch.long, device=self.device)
        self._step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids, relative_position, relative_velocity, appearances) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._tracked_position[env_ids] = relative_position[env_ids]
        self._tracked_velocity[env_ids] = relative_velocity[env_ids]
        self._appearances[env_ids] = torch.nn.functional.normalize(
            appearances[env_ids], dim=-1, eps=1.0e-8
        )
        self._valid[env_ids] = True
        self._misses[env_ids] = 0
        self._step[env_ids] = 0
        self._position_history[:, env_ids] = relative_position[env_ids].unsqueeze(0)
        self._velocity_history[:, env_ids] = relative_velocity[env_ids].unsqueeze(0)

    def step(self, relative_position, relative_velocity):
        self._tracked_position += self._tracked_velocity * float(self.cfg.control_dt)
        self._history_index = (self._history_index + 1) % self._position_history.shape[0]
        self._position_history[self._history_index] = relative_position
        self._velocity_history[self._history_index] = relative_velocity
        delayed = (self._history_index - int(self.cfg.delay_steps)) % self._position_history.shape[0]
        due = torch.remainder(self._step, int(self.cfg.detection_period_steps)) == 0
        if torch.any(due):
            env_ids = torch.nonzero(due, as_tuple=False).squeeze(-1)
            shape = (len(env_ids), ENTITY_COUNT)
            observed = torch.rand(shape, generator=self.generator, device=self.device) >= float(
                self.cfg.dropout_probability
            )
            pos_noise = torch.randn((*shape, 2), generator=self.generator, device=self.device)
            vel_noise = torch.randn((*shape, 2), generator=self.generator, device=self.device)
            measured_position = self._position_history[delayed, env_ids] + pos_noise * float(
                self.cfg.position_noise_std_m
            )
            measured_velocity = self._velocity_history[delayed, env_ids] + vel_noise * float(
                self.cfg.velocity_noise_std_mps
            )
            self._tracked_position[env_ids] = torch.where(
                observed.unsqueeze(-1), measured_position, self._tracked_position[env_ids]
            )
            self._tracked_velocity[env_ids] = torch.where(
                observed.unsqueeze(-1), measured_velocity, self._tracked_velocity[env_ids]
            )
            self._misses[env_ids] = torch.where(observed, 0, self._misses[env_ids] + 1)
            self._valid[env_ids] = self._misses[env_ids] < int(self.cfg.hold_detection_misses)
        self._step += 1
        return self.output()

    def output(self):
        raw4 = torch.cat((self._tracked_position, self._tracked_velocity, self._appearances), dim=-1)
        order = torch.argsort(
            torch.rand((self.num_envs, ENTITY_COUNT), generator=self.generator, device=self.device), dim=-1
        )
        shuffled = torch.gather(raw4, 1, order.unsqueeze(-1).expand(-1, -1, RAW_DIM))
        shuffled_valid = torch.gather(self._valid, 1, order)
        raw = torch.zeros(self.num_envs, E_MAX, RAW_DIM, device=self.device)
        mask = torch.zeros(self.num_envs, E_MAX, dtype=torch.bool, device=self.device)
        raw[:, :ENTITY_COUNT] = shuffled
        mask[:, :ENTITY_COUNT] = shuffled_valid
        return raw * mask.unsqueeze(-1), mask
