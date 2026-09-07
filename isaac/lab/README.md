# Isaac Lab T1/S2 policy environment

This is the repository's reinforcement-learning layer. It registers the
four-environment `Isaac-Semantic-ASV-T1S2-Direct-v0` task as an Isaac Lab
`DirectRLEnv` and exposes the observation, reward, reset, termination and
RSL-RL PPO configuration used to fine-tune the task-conditioned semantic
policy.

## What this environment represents

The policy action is a two-dimensional desired displacement in the vessel body
frame, capped at 0.5 m. Each environment contains the T1 target arrangement;
the red and blue targets follow the S2 trajectory. Observations contain up to
16 shuffled entities with body-frame position/velocity, a 512-dimensional
appearance feature, a validity mask and a 64-dimensional task embedding.

The Lab environment deliberately uses a fast planar kinematic response model.
It is a parallel policy-fine-tuning surrogate, not a hydrodynamic or firmware
validation. It does not run `EspApp`, inject input dither, apply propeller
forces, or claim controller acceptance.

For controller and HIL validation, use
[`../examples/floating_asv.py`](../examples/floating_asv.py). That path applies
reversible forces at the two physical propellers to the full six-degree-of-
freedom rigid body with buoyancy, added mass and drag.

## Runtime assets

Three runtime artifacts are required but intentionally excluded from Git:

- `runtime_assets/owl512_appearance_library_v1.npz`
- `runtime_assets/qwen_task_embed.npz`
- `runtime_assets/actor_bc_semantic16_v13_dagger_polar_frozen.pt` (PPO only)

The first two are required by the smoke task. Paths may be overridden with
`ASV_LAB_APPEARANCE_LIBRARY`, `ASV_LAB_TASK_EMBEDDINGS` and
`ASV_LAB_ACTOR_CHECKPOINT`. Keeping them under `runtime_assets/` makes a local
workspace self-contained without publishing model/data artifacts.

## Install and smoke

Run with the Python supplied by the same Isaac Sim installation used by Isaac
Lab:

```bat
cd isaac\lab
E:\isaacsim\_build\windows-x86_64\release\python.bat -m pip install -e .
E:\isaacsim\_build\windows-x86_64\release\python.bat scripts\smoke_env.py --viz none --steps 40
```

A successful run prints `ISAACLAB_ASV_4ENV_PASS` and an observation shape of
`(4, 8336)`.

## PPO fine-tuning

After installing this package, use the thin wrapper that registers the external
ASV task and then invokes Isaac Lab's stock RSL-RL launcher:

```bat
E:\isaacsim\_build\windows-x86_64\release\python.bat scripts\train.py --task Isaac-Semantic-ASV-T1S2-Direct-v0 --viz none
```

The actor checkpoint initializes the semantic projector/selector and polar
control head. The default profile freezes the selector and projector and
updates only the polar control head plus exploration variance. Conservative
and deployment-safe runner configurations are also registered.
