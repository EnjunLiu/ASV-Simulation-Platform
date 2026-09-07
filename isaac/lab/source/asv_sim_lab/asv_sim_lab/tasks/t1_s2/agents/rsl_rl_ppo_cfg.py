"""RSL-RL PPO profiles for semantic ASV fine-tuning."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from ..assets import ACTOR_CHECKPOINT


@configclass
class SemanticActorCfg:
    class_name: str = "asv_sim_lab.tasks.t1_s2.rsl_model:SemanticPpoActor"
    checkpoint_path: str = ACTOR_CHECKPOINT
    distribution_cfg: RslRlMLPModelCfg.GaussianDistributionCfg = (
        RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.04, std_type="log")
    )


def _critic():
    return RslRlMLPModelCfg(
        hidden_dims=[128, 64], activation="elu", obs_normalization=False, distribution_cfg=None
    )


@configclass
class SemanticAsvPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 200
    save_interval = 25
    experiment_name = "semantic_asv_t1_s2_ppo"
    seed = 20260905
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SemanticActorCfg()
    critic = _critic()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.15, entropy_coef=0.001,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=2.0e-4, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=0.5
    )


@configclass
class SemanticAsvConservativePpoRunnerCfg(SemanticAsvPpoRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 40
    save_interval = 5
    experiment_name = "semantic_asv_t1_s2_ppo_conservative"
    actor = SemanticActorCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.015, std_type="log")
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.08, entropy_coef=0.0001,
        num_learning_epochs=3, num_mini_batches=4, learning_rate=2.0e-5, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.003, max_grad_norm=0.25
    )


@configclass
class SemanticAsvDeploymentSafePpoRunnerCfg(SemanticAsvPpoRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 8
    save_interval = 1
    experiment_name = "semantic_asv_t1_s2_ppo_deployment_safe"
    actor = SemanticActorCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.005, std_type="log")
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.03, entropy_coef=0.0,
        num_learning_epochs=1, num_mini_batches=4, learning_rate=5.0e-7, schedule="fixed",
        gamma=0.99, lam=0.95, desired_kl=0.001, max_grad_norm=0.1
    )
