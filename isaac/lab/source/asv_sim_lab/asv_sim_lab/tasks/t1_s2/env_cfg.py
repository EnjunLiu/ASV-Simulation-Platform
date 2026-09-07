"""Isaac Lab configuration for four-environment T1/S2 fine-tuning."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from .assets import APPEARANCE_LIBRARY, TASK_EMBEDDINGS
from .observation import POLICY_OBS_DIM


@configclass
class SemanticAsvEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 30.0
    action_space = 2
    observation_space = POLICY_OBS_DIM
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.1, render_interval=1, physics=PhysxCfg())
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=20.0, replicate_physics=True, clone_in_fabric=True
    )
    appearance_library: str = APPEARANCE_LIBRARY
    task_embeddings: str = TASK_EMBEDDINGS
    seed: int = 20260905
    collision_range_m: float = 1.0
    lost_range_m: float = 15.0
    lost_bearing_rad: float = 1.3962634015954636
