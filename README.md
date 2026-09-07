# ASV Simulation Platform

Physics-first and visual simulation environments for autonomous surface-vessel
research. The repository contains two independent backends that share the same
project-owned vessel assets and external-control boundary.

## Backends

- [`isaac/`](isaac/README.md) — Isaac Sim full-dynamics HIL with mesh buoyancy,
  added mass, drag, waves, sensors and twin-propeller actuation, plus a separate
  four-environment Isaac Lab `DirectRLEnv` for T1/S2 policy fine-tuning.
- [`unreal/`](unreal/README.md) — Unreal Engine hardware-in-the-loop environment
  with ocean rendering, deterministic scene automation, camera emulation and a
  TCP interface for an external Jetson runtime.

Blender and USD are used as asset-authoring and interchange tools. They are not
presented as a third simulation backend.

Each backend has its own setup and run instructions. This repository contains
no deployed autonomy runtime, model weights, datasets, experiment logs or
low-level controller implementation. The Isaac Lab task and PPO configuration
are source code; private runtime artifacts are excluded.
