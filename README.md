# ASV Simulation Platform

Physics-first and visual simulation environments for autonomous surface-vessel
research. The repository contains two independent backends that share the same
project-owned vessel assets and external-control boundary.

## Backends

- [`isaac/`](isaac/README.md) — Isaac Sim / Isaac Lab environment with a full
  six-degree-of-freedom rigid body, mesh buoyancy, added mass, quadratic drag,
  waves, sensor simulation and reversible twin-propeller actuation.
- [`unreal/`](unreal/README.md) — Unreal Engine hardware-in-the-loop environment
  with ocean rendering, deterministic scene automation, camera emulation and a
  TCP interface for an external Jetson runtime.

Blender and USD are used as asset-authoring and interchange tools. They are not
presented as a third simulation backend.

Each backend has its own setup and run instructions. This repository contains
no autonomy runtime, training pipeline, model weights, datasets, experiment logs
or low-level controller implementation.
