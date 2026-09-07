"""Registration for the T1/S2 semantic ASV task."""

from __future__ import annotations

import gymnasium as gym

TASK_ID = "Isaac-Semantic-ASV-T1S2-Direct-v0"


def register_task() -> None:
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point=f"{__name__}.env:SemanticAsvEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:SemanticAsvEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SemanticAsvPpoRunnerCfg",
            "rsl_rl_conservative_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:SemanticAsvConservativePpoRunnerCfg"
            ),
            "rsl_rl_deployment_safe_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:SemanticAsvDeploymentSafePpoRunnerCfg"
            ),
        },
    )


register_task()

__all__ = ["TASK_ID", "register_task"]
