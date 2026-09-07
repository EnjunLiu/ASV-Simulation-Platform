"""RSL-RL actor initialized from the semantic behavior-cloning checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from .observation import E_MAX, POLICY_OBS_DIM, RAW_DIM, TASK_DIM
from .semantic_actor import load_semantic_actor

RAW_FLAT_DIM = E_MAX * RAW_DIM


def unpack_policy_observation(policy: torch.Tensor):
    if policy.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(f"policy observation expected {POLICY_OBS_DIM}, got {policy.shape[-1]}")
    raw = policy[..., :RAW_FLAT_DIM].reshape(*policy.shape[:-1], E_MAX, RAW_DIM)
    mask = policy[..., RAW_FLAT_DIM : RAW_FLAT_DIM + E_MAX] > 0.5
    return raw, mask, policy[..., -TASK_DIM:]


class SemanticPpoActor(nn.Module):
    """Train only the polar control head and Gaussian exploration variance."""

    is_recurrent = False

    def __init__(self, obs: TensorDict, obs_groups, obs_set, output_dim, checkpoint_path,
                 init_std=0.04, std_type="log", distribution_cfg=None) -> None:
        super().__init__()
        del obs
        if int(output_dim) != 2 or obs_groups[obs_set] != ["policy"]:
            raise ValueError("semantic actor requires policy observations and two actions")
        self.obs_groups = obs_groups[obs_set]
        self.semantic = load_semantic_actor(checkpoint_path, map_location="cpu")
        for parameter in self.semantic.parameters():
            parameter.requires_grad_(False)
        for parameter in self.semantic.polar_control.parameters():
            parameter.requires_grad_(True)
        if distribution_cfg:
            init_std = float(distribution_cfg.get("init_std", init_std))
            std_type = str(distribution_cfg.get("std_type", std_type))
        self.distribution = GaussianDistribution(output_dim, init_std=init_std, std_type=std_type)

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False):
        del masks, hidden_state
        raw, valid, task = unpack_policy_observation(obs["policy"])
        mean = self.semantic(raw, task, valid)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def reset(self, dones=None, hidden_state=None):
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        del dones

    def update_normalization(self, obs):
        del obs

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.std

    @property
    def output_entropy(self):
        return self.distribution.entropy

    @property
    def output_distribution_params(self):
        return self.distribution.params

    def get_output_log_prob(self, outputs):
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)
