"""Launch and step the four-environment Isaac Lab ASV task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import asv_sim_lab  # noqa: E402, F401
from asv_sim_lab.tasks.t1_s2 import TASK_ID  # noqa: E402
from asv_sim_lab.tasks.t1_s2.env_cfg import SemanticAsvEnvCfg  # noqa: E402

cfg = SemanticAsvEnvCfg()
cfg.scene.num_envs = 4
env = gym.make(TASK_ID, cfg=cfg)
observation, _ = env.reset()
assert observation["policy"].shape == (4, cfg.observation_space)
for _ in range(int(args.steps)):
    actions = torch.zeros((4, 2), device=env.unwrapped.device)
    observation, rewards, terminated, truncated, _ = env.step(actions)
assert torch.isfinite(observation["policy"]).all()
assert torch.isfinite(rewards).all()
print(
    "ISAACLAB_ASV_4ENV_PASS",
    "obs",
    tuple(observation["policy"].shape),
    "reward_mean",
    float(rewards.mean()),
    "terminated",
    int(terminated.sum()),
    "truncated",
    int(truncated.sum()),
)
env.close()
simulation_app.close()
