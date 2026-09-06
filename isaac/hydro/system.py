from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .forces import hydro_force_on_box
from .water import WaterState


def _as_numpy(arr) -> np.ndarray:
    if arr is None:
        return np.zeros(0)
    if hasattr(arr, "numpy"):
        return np.asarray(arr.numpy())
    return np.asarray(arr)


@dataclass
class RigidState:
    position: np.ndarray
    orientation: np.ndarray
    linear: np.ndarray
    angular: np.ndarray
    mass: float | None = None


def read_rigid_state(rigid_prim) -> RigidState | None:
    if hasattr(rigid_prim, "is_physics_tensor_entity_valid"):
        if not rigid_prim.is_physics_tensor_entity_valid():
            return None
    positions, orientations = rigid_prim.get_world_poses()
    linear, angular = rigid_prim.get_velocities()
    positions = _as_numpy(positions).reshape(-1, 3)
    orientations = _as_numpy(orientations).reshape(-1, 4)
    linear = _as_numpy(linear).reshape(-1, 3)
    angular = _as_numpy(angular).reshape(-1, 3)
    if positions.shape[0] == 0:
        return None
    mass = None
    if hasattr(rigid_prim, "get_masses"):
        masses = _as_numpy(rigid_prim.get_masses()).reshape(-1)
        if masses.size:
            mass = float(masses[0])
    return RigidState(
        position=positions[0],
        orientation=orientations[0],
        linear=linear[0],
        angular=angular[0],
        mass=mass,
    )


@dataclass
class HydroBody:
    half_extents: np.ndarray | None = None
    cd: float = 1.0
    angular_damping: float = 6.0
    heave_damping_ratio: float = 0.45
    horizontal_damping_ratio: float = 0.25
    surge_damping_ratio: float | None = None
    sway_damping_ratio: float | None = None
    mass: float = 400.0
    volume_filt: float | None = None
    local_center: np.ndarray | None = None
    mesh: object | None = None
    cm: float = 1.0
    cb: float = 1.0

    def __post_init__(self) -> None:
        if self.mesh is not None:
            if self.half_extents is None:
                self.half_extents = np.asarray(self.mesh.half_extents, dtype=np.float64)
            if self.local_center is None:
                self.local_center = np.asarray(self.mesh.aabb_center, dtype=np.float64)
        elif self.half_extents is None:
            raise ValueError("HydroBody needs mesh or half_extents")
        self.half_extents = np.asarray(self.half_extents, dtype=np.float64).reshape(3)


@dataclass
class HydroItem:
    rigid_prim: object
    body: HydroBody
    path: str = ""


@dataclass
class HydroSystem:
    water: WaterState
    items: list[HydroItem] = field(default_factory=list)
    last_debug: list[dict] = field(default_factory=list)

    def add(self, rigid_prim: object, body: HydroBody, path: str = "") -> None:
        self.items.append(HydroItem(rigid_prim=rigid_prim, body=body, path=path))

    def clear(self) -> None:
        self.items.clear()
        self.last_debug = []

    @classmethod
    def from_stage(cls, stage, water: WaterState | None = None, skip_substrings=None) -> "HydroSystem":
        from isaacsim.core.experimental.prims import RigidPrim

        from .discover import describe_hydro_bodies

        hydro = cls(water=water or WaterState())
        for desc in describe_hydro_bodies(stage, skip_substrings):
            prim = RigidPrim(paths=desc["path"], masses=[desc["mass"]])
            body = HydroBody(half_extents=np.asarray(desc["half_extents"]), mass=float(desc["mass"]))
            hydro.add(prim, body, path=desc["path"])
        return hydro

    def _step_one(self, rigid_prim, body: HydroBody, state: RigidState | None = None) -> dict | None:
        if state is None:
            state = read_rigid_state(rigid_prim)
        if state is None:
            return None
        mass = float(state.mass) if state.mass is not None else body.mass
        f, t, p, info = hydro_force_on_box(
            state.position,
            state.orientation,
            body.half_extents,
            state.linear,
            state.angular,
            self.water,
            mass=mass,
            cd=body.cd,
            angular_damping=body.angular_damping,
            heave_damping_ratio=body.heave_damping_ratio,
            horizontal_damping_ratio=body.horizontal_damping_ratio,
            surge_damping_ratio=body.surge_damping_ratio,
            sway_damping_ratio=body.sway_damping_ratio,
            volume_filt=body.volume_filt,
            local_center=body.local_center,
            mesh=body.mesh,
            cm=body.cm,
            cb=body.cb,
        )
        body.volume_filt = float(info["volume"])
        rigid_prim.apply_forces_and_torques_at_pos(
            np.asarray(f, dtype=np.float32).reshape(1, 3),
            np.asarray(t, dtype=np.float32).reshape(1, 3),
            positions=np.asarray(p, dtype=np.float32).reshape(1, 3),
        )
        info["path"] = getattr(rigid_prim, "paths", None)
        return info

    def physics_step(self, dt: float, context: object = None, states: list[RigidState | None] | None = None) -> None:
        debug = []
        for i, item in enumerate(self.items):
            st = states[i] if states is not None and i < len(states) else None
            info = self._step_one(item.rigid_prim, item.body, state=st)
            if info is not None:
                info["path"] = item.path
                debug.append(info)
        self.last_debug = debug
