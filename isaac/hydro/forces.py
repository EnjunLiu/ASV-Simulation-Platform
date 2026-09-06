from __future__ import annotations

import numpy as np

from .volume import box_world_aabb, sampled_submerged
from .volume import quat_wxyz_to_rot
from .water import WaterState


def hydro_force_on_box(
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    half_extents: np.ndarray | None,
    linear_vel: np.ndarray,
    angular_vel: np.ndarray,
    water: WaterState,
    *,
    mass: float = 400.0,
    cd: float = 1.0,
    angular_damping: float = 6.0,
    heave_damping_ratio: float = 0.45,
    horizontal_damping_ratio: float = 0.25,
    surge_damping_ratio: float | None = None,
    sway_damping_ratio: float | None = None,
    volume_filt: float | None = None,
    local_center: np.ndarray | None = None,
    mesh=None,
    cm: float = 1.0,
    cb: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Buoyancy at CoB, quadratic drag, viscous heave damping, angular damping.

    Near the waterline buoyancy is a stiff spring (k = rho*g*A). Quadratic drag
    vanishes as v->0, so an extra linear heave damper (~critical) is required
    or the body will bounce forever.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    lin = np.asarray(linear_vel, dtype=np.float64).reshape(3)
    ang = np.asarray(angular_vel, dtype=np.float64).reshape(3)

    u_water = np.zeros(3, dtype=np.float64)
    a_water = np.zeros(3, dtype=np.float64)
    area_used = np.zeros(3, dtype=np.float64)
    inertia = np.zeros(3, dtype=np.float64)

    if mesh is not None:
        sub = mesh.submerged(center, quat_wxyz, water)
        volume = float(sub.volume)
        cob = np.asarray(sub.cob, dtype=np.float64).reshape(3)
        wp_area = float(sub.waterplane_area)
        water_z = float(sub.water_z)
        sub_h = float(sub.sub_h)
        mn, mx = mesh.world_aabb(center, quat_wxyz)
        full_vol = max(float(mesh.volume), 1e-9)
        frac = float(np.clip(volume / full_vol, 0.0, 1.0))
        area_xy = wp_area
        u_water = np.asarray(sub.u_water, dtype=np.float64).reshape(3)
        a_water = np.asarray(sub.a_water, dtype=np.float64).reshape(3)
    else:
        half = np.asarray(half_extents, dtype=np.float64).reshape(3)
        mn, mx = box_world_aabb(center, quat_wxyz, half, local_center=local_center)
        nx, ny = (6, 3) if getattr(water, "waves", None) is not None else (1, 1)
        volume, cob, sub_h, water_z = sampled_submerged(mn, mx, water.sample_height, nx=nx, ny=ny)
        full_h = max(1e-9, float(mx[2] - mn[2]))
        frac = float(np.clip(sub_h / full_h, 0.0, 1.0))
        area_xy = max(1e-9, float((mx[0] - mn[0]) * (mx[1] - mn[1])))
        wp_area = area_xy

    if volume_filt is None:
        volume_used = volume
    else:
        volume_used = 0.65 * float(volume_filt) + 0.35 * volume

    buoyancy = np.array(
        [0.0, 0.0, water.rho * water.gravity * float(cb) * volume_used],
        dtype=np.float64,
    )
    v_rel = lin - u_water

    drag_q = np.zeros(3, dtype=np.float64)
    if volume_used > 0.0:
        if mesh is not None:
            area_used = _facing_area(sub.area_pos, sub.area_neg, v_rel)
            if wp_area > 1e-8:
                area_used[2] = wp_area
            for i in range(3):
                vi = float(v_rel[i])
                if abs(vi) > 1e-8 and area_used[i] > 0.0:
                    drag_q[i] = -0.5 * water.rho * cd * area_used[i] * abs(vi) * vi
        else:
            area = area_xy * max(frac, 0.05 if volume_used > 0 else 0.0)
            speed = float(np.linalg.norm(v_rel))
            if speed > 1e-8:
                drag_q = -0.5 * water.rho * cd * area * speed * v_rel
            area_used[:] = area

    # Critical heave damping: 2*zeta*sqrt(k m), k = rho g A (waterplane stiffness)
    drag_lin = np.zeros(3, dtype=np.float64)
    if volume_used > 0.0 and mass > 1e-6 and area_xy > 1e-8:
        k_heave = water.rho * water.gravity * area_xy
        b_crit = 2.0 * np.sqrt(max(k_heave * mass, 0.0))
        b = heave_damping_ratio * b_crit
        drag_lin[2] = -b * v_rel[2]
        if surge_damping_ratio is None and sway_damping_ratio is None:
            # Preserve the historical isotropic world-horizontal behavior.
            drag_lin[0] = -float(horizontal_damping_ratio) * b * v_rel[0]
            drag_lin[1] = -float(horizontal_damping_ratio) * b * v_rel[1]
        else:
            surge = float(horizontal_damping_ratio if surge_damping_ratio is None else surge_damping_ratio)
            sway = float(horizontal_damping_ratio if sway_damping_ratio is None else sway_damping_ratio)
            rotation = quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
            body_velocity = rotation.T @ v_rel
            body_damping = np.array(
                [-surge * b * body_velocity[0], -sway * b * body_velocity[1], 0.0],
                dtype=np.float64,
            )
            drag_lin += rotation @ body_damping

    if mesh is not None and volume_used > 0.0 and abs(float(cm)) > 1e-12:
        inertia = water.rho * float(cm) * volume_used * a_water

    force = buoyancy + drag_q + drag_lin + inertia
    torque = -angular_damping * ang
    debug = {
        "volume": volume_used,
        "volume_raw": volume,
        "frac": frac,
        "buoyancy_z": float(buoyancy[2]),
        "drag": drag_q + drag_lin,
        "inertia": inertia,
        "cob": cob,
        "aabb_min": mn,
        "aabb_max": mx,
        "water_z": water_z,
        "waterplane_area": float(wp_area),
        "area": area_used,
        "u_water": u_water,
        "a_water": a_water,
        "cb": float(cb),
        "hull_volume": float(getattr(mesh, "volume", 0.0)) if mesh is not None else float((mx[0] - mn[0]) * (mx[1] - mn[1]) * (mx[2] - mn[2])),
    }
    return force, torque, cob, debug


def _facing_area(area_pos, area_neg, v_rel) -> np.ndarray:
    pos = np.asarray(area_pos, dtype=np.float64).reshape(3)
    neg = np.asarray(area_neg, dtype=np.float64).reshape(3)
    v = np.asarray(v_rel, dtype=np.float64).reshape(3)
    out = np.zeros(3, dtype=np.float64)
    for i in range(3):
        out[i] = float(pos[i] if v[i] >= 0.0 else neg[i])
    return out
