"""Point-force thrusters in a rigid-body frame (ASV propellers)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .volume import quat_wxyz_to_rot


def _as_numpy(arr) -> np.ndarray:
    if arr is None:
        return np.zeros(0)
    if hasattr(arr, "numpy"):
        return np.asarray(arr.numpy())
    return np.asarray(arr)


@dataclass
class Thruster:
    """Force along local_dir at local_pos, both in the rigid body frame."""

    name: str
    local_pos: np.ndarray
    local_dir: np.ndarray
    command: float = 0.0
    max_force: float = 40.0

    def force_and_pos(self, body_pos: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = quat_wxyz_to_rot(quat_wxyz)
        d = np.asarray(self.local_dir, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return np.zeros(3), np.asarray(body_pos, dtype=np.float64).reshape(3)
        d = d / n
        mag = float(np.clip(self.command, -self.max_force, self.max_force))
        force = mag * (R @ d)
        pos = np.asarray(body_pos, dtype=np.float64).reshape(3) + R @ np.asarray(self.local_pos, dtype=np.float64).reshape(3)
        return force, pos


@dataclass
class ThrusterBank:
    thrusters: list[Thruster] = field(default_factory=list)

    def set_commands(self, mapping: dict[str, float]) -> None:
        for t in self.thrusters:
            if t.name in mapping:
                t.command = float(mapping[t.name])

    def physics_step(self, rigid_prim, state=None) -> dict:
        if state is not None:
            body_pos = np.asarray(state.position, dtype=np.float64).reshape(3)
            quat = np.asarray(state.orientation, dtype=np.float64).reshape(4)
        else:
            if hasattr(rigid_prim, "is_physics_tensor_entity_valid"):
                if not rigid_prim.is_physics_tensor_entity_valid():
                    return {}
            positions, orientations = rigid_prim.get_world_poses()
            positions = _as_numpy(positions).reshape(-1, 3)
            orientations = _as_numpy(orientations).reshape(-1, 4)
            if positions.shape[0] == 0:
                return {}
            body_pos = positions[0]
            quat = orientations[0]
        total_f = np.zeros(3, dtype=np.float64)
        total_tau = np.zeros(3, dtype=np.float64)
        debug = {}
        for t in self.thrusters:
            f, p = t.force_and_pos(body_pos, quat)
            r = p - body_pos
            total_f = total_f + f
            total_tau = total_tau + np.cross(r, f)
            debug[t.name] = float(np.clip(t.command, -t.max_force, t.max_force))
        if float(np.linalg.norm(total_f)) > 1e-9 or float(np.linalg.norm(total_tau)) > 1e-9:
            rigid_prim.apply_forces_and_torques_at_pos(
                np.asarray(total_f, dtype=np.float32).reshape(1, 3),
                np.asarray(total_tau, dtype=np.float32).reshape(1, 3),
                positions=np.asarray(body_pos, dtype=np.float32).reshape(1, 3),
            )
        debug["force"] = total_f
        debug["torque"] = total_tau
        return debug
