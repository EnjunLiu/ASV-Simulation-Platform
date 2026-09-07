# ASV Simulation Platform — Isaac Sim / Isaac Lab

Physics-first simulation environment for an autonomous surface vessel. The
platform models a six-degree-of-freedom rigid hull, mesh-based buoyancy,
added-mass and quadratic hydrodynamic forces, waves, damping and reversible
twin propellers applied at their physical locations.

This directory is intentionally simulation-only. It contains no deployed VLA
runtime, model weights, datasets, acceptance logs or low-level controller
implementation. The `lab/` subproject contains the public Isaac Lab task used
for policy fine-tuning; its runtime artifacts remain local and Git-ignored.

## Contents

- `assets/asv_models/`: USD and Blender source assets, materials, variants and HDR lighting.
- `hydro/`: reusable buoyancy, hydrodynamics, ocean, wake, sensor and thruster modules.
- `examples/floating_asv.py`: complete ASV scene with manual thrust or external wrench actuation.
- `examples/floating_cube.py` and `examples/multi_body.py`: compact buoyancy demonstrations.
- `exts/`: Isaac extension integration.
- `config/`: Fast DDS configuration for distributed ROS 2 simulation.
- `lab/`: installable Isaac Lab `DirectRLEnv`, T1/S2 task model, noisy semantic
  observations and RSL-RL PPO configuration.

## Run

```bat
E:\isaacsim\_build\windows-x86_64\release\python.bat E:\hil-platform\isaac\examples\floating_asv.py
```

Useful modes:

```bat
:: Three-second headless physics smoke
E:\isaacsim\_build\windows-x86_64\release\python.bat E:\hil-platform\isaac\examples\floating_asv.py --headless --smoke --no-ros

:: Static water, T1/S2 targets, external controller force/moment
E:\isaacsim\_build\windows-x86_64\release\python.bat E:\hil-platform\isaac\examples\floating_asv.py --layout T1 --motion S2 --seed 0 --amp 0 --actuation wrench --thrust 0
```

In `wrench` mode Isaac subscribes to `/asv/control_wrench`, validates freshness,
and maps total surge force `F` and yaw moment `N` to the two reversible
propellers. No controller or guidance law runs inside Isaac. The simulator
publishes `/clock`, `/asv/odom`, `/asv/imu`, `/asv/gnss/fix`, camera images and
target odometry for external modules.

The main physical parameters are exposed as command-line options: hull mass,
buoyancy coefficient, drag and added-mass coefficients, heave/horizontal/angular
damping, propeller spacing and per-propeller thrust limit.

## Isaac Lab policy environment

[`lab/`](lab/README.md) is a separate fast policy-iteration path. It runs four
parallel T1/S2 environments and preserves the deployment-shaped semantic
observation, reward, termination/reset logic and PPO profiles. Its planar
kinematic response is intentionally a throughput-oriented surrogate. It is not
used as evidence for hydrodynamics, propeller actuation or controller
acceptance; those claims require `floating_asv.py` in full-dynamics wrench mode.

## Assets and portability

All project-owned assets are relative to this directory. Moving the complete
`isaac` folder does not require rewriting paths. Isaac Sim itself remains an
external installation.
