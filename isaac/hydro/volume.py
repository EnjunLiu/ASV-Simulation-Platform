"""Submerged volume of a box vs a still water plane.

Phase 1 uses the world-axis AABB of the rotated box, then clips that AABB
against z <= water_z. Exact for an axis-aligned cube; slightly conservative
when the cube is tilted.
"""

from __future__ import annotations

import numpy as np

_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=np.float64,
)


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def box_world_aabb(
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    half_extents: np.ndarray,
    local_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    half = np.asarray(half_extents, dtype=np.float64).reshape(3)
    R = quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    offset = np.zeros(3, dtype=np.float64) if local_center is None else np.asarray(local_center, dtype=np.float64).reshape(3)
    world_center = center + R @ offset
    corners = (_CORNER_SIGNS * half) @ R.T + world_center
    return corners.min(axis=0), corners.max(axis=0)


def aabb_submerged(mn: np.ndarray, mx: np.ndarray, water_z: float) -> tuple[float, np.ndarray, float]:
    """Return (volume, center_of_buoyancy, submerged_height)."""
    mn = np.asarray(mn, dtype=np.float64).reshape(3)
    mx = np.asarray(mx, dtype=np.float64).reshape(3)
    dx = max(0.0, float(mx[0] - mn[0]))
    dy = max(0.0, float(mx[1] - mn[1]))
    z0, z1 = float(mn[2]), float(mx[2])
    height = max(0.0, z1 - z0)
    if height <= 0.0 or dx <= 0.0 or dy <= 0.0:
        cob = 0.5 * (mn + mx)
        return 0.0, cob, 0.0
    if z1 <= water_z:
        v = dx * dy * height
        return v, 0.5 * (mn + mx), height
    if z0 >= water_z:
        cob = 0.5 * (mn + mx)
        return 0.0, cob, 0.0
    h = water_z - z0
    v = dx * dy * h
    cob = np.array([(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, z0 + 0.5 * h], dtype=np.float64)
    return v, cob, h


def sampled_submerged(mn, mx, sample_h, nx: int = 6, ny: int = 3):
    """Clip an AABB against a spatially varying water height.

    sample_h(x, y) -> water_z. Returns (volume, cob, mean_sub_h, mean_water_z).
    """
    mn = np.asarray(mn, dtype=np.float64).reshape(3)
    mx = np.asarray(mx, dtype=np.float64).reshape(3)
    nx = max(int(nx), 1)
    ny = max(int(ny), 1)
    dx = (mx[0] - mn[0]) / nx
    dy = (mx[1] - mn[1]) / ny
    volume = 0.0
    cob_acc = np.zeros(3, dtype=np.float64)
    sub_acc = 0.0
    z_acc = 0.0
    n_cells = 0
    for i in range(nx):
        x0 = mn[0] + i * dx
        x1 = x0 + dx
        cx = 0.5 * (x0 + x1)
        for j in range(ny):
            y0 = mn[1] + j * dy
            y1 = y0 + dy
            cy = 0.5 * (y0 + y1)
            water_z = float(sample_h(cx, cy))
            v, cob, h = aabb_submerged([x0, y0, mn[2]], [x1, y1, mx[2]], water_z)
            volume += v
            cob_acc = cob_acc + cob * v
            sub_acc += h
            z_acc += water_z
            n_cells += 1
    mean_h = sub_acc / max(n_cells, 1)
    mean_z = z_acc / max(n_cells, 1)
    if volume <= 1e-18:
        return 0.0, 0.5 * (mn + mx), mean_h, mean_z
    return volume, cob_acc / volume, mean_h, mean_z
