"""Find dynamic rigid bodies on a USD stage and estimate box half-extents."""

from __future__ import annotations

import numpy as np

_SKIP_SUBSTR = (
    "/Water",
    "/Ground",
    "/Floor",
    "/DistantLight",
    "/Sky",
    "/Sun",
    "/Camera",
    "/OmniverseKit",
    "/Render",
    "/Environment",
)


def _scale_from_world_xform(m) -> np.ndarray:
    a = np.array(m, dtype=np.float64).reshape(4, 4)
    sx = float(np.linalg.norm(a[0:3, 0]))
    sy = float(np.linalg.norm(a[0:3, 1]))
    sz = float(np.linalg.norm(a[0:3, 2]))
    return np.array([max(sx, 1e-6), max(sy, 1e-6), max(sz, 1e-6)])


def prim_local_half_extents(prim) -> np.ndarray | None:
    from pxr import Usd, UsdGeom

    half = None
    if prim.IsA(UsdGeom.Cube):
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 1.0)
        half = np.array([0.5 * size, 0.5 * size, 0.5 * size], dtype=np.float64)
    elif prim.IsA(UsdGeom.Sphere):
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 0.5)
        half = np.array([r, r, r], dtype=np.float64)
    elif prim.IsA(UsdGeom.Capsule):
        cap = UsdGeom.Capsule(prim)
        r = float(cap.GetRadiusAttr().Get() or 0.25)
        h = float(cap.GetHeightAttr().Get() or 1.0)
        half = np.array([r, r, 0.5 * h + r], dtype=np.float64)
    boundable = UsdGeom.Boundable(prim) if prim.IsA(UsdGeom.Boundable) else None
    if half is None and boundable is not None:
        extent = boundable.GetExtentAttr().Get()
        if extent and len(extent) == 2:
            a, b = extent
            half = 0.5 * np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], dtype=np.float64)
    if half is None:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeLocalBound(prim).GetRange()
        if rng.IsEmpty():
            return None
        a, b = rng.GetMin(), rng.GetMax()
        half = 0.5 * np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], dtype=np.float64)
    if half is None or np.min(np.abs(half)) <= 1e-8:
        return None
    xformable = UsdGeom.Xformable(prim)
    world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return half * _scale_from_world_xform(world)


def prim_mass(prim, half: np.ndarray, density: float = 400.0) -> float:
    from pxr import UsdPhysics

    if prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(prim)
        m = mass_api.GetMassAttr().Get()
        if m and float(m) > 1e-6:
            return float(m)
        d = mass_api.GetDensityAttr().Get()
        if d and float(d) > 1e-6:
            vol = float(8.0 * half[0] * half[1] * half[2])
            return float(d) * vol
    vol = float(8.0 * half[0] * half[1] * half[2])
    return max(density * vol, 1e-3)


def iter_dynamic_rigid_prims(stage, skip_substrings=None):
    if skip_substrings is None:
        skip_substrings = _SKIP_SUBSTR
    from pxr import UsdPhysics

    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rb = UsdPhysics.RigidBodyAPI(prim)
        kin = rb.GetKinematicEnabledAttr().Get()
        if kin:
            continue
        path = str(prim.GetPath())
        if any(s in path for s in skip_substrings):
            continue
        yield prim


def describe_hydro_bodies(stage, skip_substrings=None) -> list[dict]:
    if skip_substrings is None:
        skip_substrings = _SKIP_SUBSTR
    found = []
    for prim in iter_dynamic_rigid_prims(stage, skip_substrings):
        half = prim_local_half_extents(prim)
        if half is None or np.min(half) <= 1e-6:
            continue
        path = str(prim.GetPath())
        found.append(
            {
                "path": path,
                "half_extents": half,
                "mass": prim_mass(prim, half),
            }
        )
    return found


def aabb_in_frame(target_prim, frame_prim) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis-aligned box of target_prim expressed in frame_prim local coordinates.

    Returns (center, half_extents) or None.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    world_bound = cache.ComputeWorldBound(target_prim)
    rng = world_bound.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn_w = np.array(rng.GetMin(), dtype=np.float64)
    mx_w = np.array(rng.GetMax(), dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x in (mn_w[0], mx_w[0]) for y in (mn_w[1], mx_w[1]) for z in (mn_w[2], mx_w[2])],
        dtype=np.float64,
    )
    frame = np.array(UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).reshape(4, 4)
    inv = np.linalg.inv(frame)
    homog = np.concatenate([corners, np.ones((8, 1))], axis=1)
    local = (inv @ homog.T).T[:, :3]
    mn, mx = local.min(axis=0), local.max(axis=0)
    center = 0.5 * (mn + mx)
    half = 0.5 * (mx - mn)
    if np.min(half) <= 1e-8:
        return None
    return center, half


def origin_in_frame(target_prim, frame_prim) -> np.ndarray | None:
    """Origin of target_prim in frame_prim local coordinates."""
    from pxr import Usd, UsdGeom

    if target_prim is None or not target_prim.IsValid():
        return None
    tw = np.array(UsdGeom.Xformable(target_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).reshape(4, 4)
    fw = np.array(UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).reshape(4, 4)
    local = np.linalg.inv(fw) @ tw
    return local[:3, 3].copy()
