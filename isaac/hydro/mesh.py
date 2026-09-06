"""Closed triangle mesh: enclosed volume and clip-against-water submerged volume."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .volume import quat_wxyz_to_rot


@dataclass
class SubmergedInfo:
    volume: float
    cob: np.ndarray
    waterplane_area: float
    water_z: float
    sub_h: float
    area_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    area_neg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    u_water: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    a_water: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


def triangulate_faces(counts, indices) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int32).reshape(-1)
    indices = np.asarray(indices, dtype=np.int32).reshape(-1)
    faces = []
    cursor = 0
    for c in counts:
        c = int(c)
        poly = indices[cursor : cursor + c]
        cursor += c
        if c < 3:
            continue
        for k in range(1, c - 1):
            faces.append((int(poly[0]), int(poly[k]), int(poly[k + 1])))
    if not faces:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(faces, dtype=np.int32)


def signed_volume_and_centroid(verts: np.ndarray, faces: np.ndarray) -> tuple[float, np.ndarray]:
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if faces.shape[0] == 0:
        return 0.0, np.zeros(3, dtype=np.float64)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1, v2)
    vol6 = np.einsum("ij,ij->i", v0, cross)
    volume = float(vol6.sum() / 6.0)
    if abs(volume) < 1e-18:
        return 0.0, verts.mean(axis=0)
    cob = ((v0 + v1 + v2) * vol6[:, None]).sum(axis=0) / (24.0 * volume)
    return volume, cob


def edge_watertight(faces: np.ndarray) -> bool:
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if faces.shape[0] == 0:
        return False
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for e in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edges[key] = edges.get(key, 0) + 1
    return bool(edges) and all(v == 2 for v in edges.values())


def box_hull_mesh(half_extents, center=None) -> "HullMesh":
    """Axis-aligned closed box in the given frame (12 triangles)."""
    half = np.asarray(half_extents, dtype=np.float64).reshape(3)
    c = np.zeros(3, dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64).reshape(3)
    hx, hy, hz = [float(v) for v in half]
    verts = np.array(
        [
            [c[0] - hx, c[1] - hy, c[2] - hz],
            [c[0] + hx, c[1] - hy, c[2] - hz],
            [c[0] + hx, c[1] + hy, c[2] - hz],
            [c[0] - hx, c[1] + hy, c[2] - hz],
            [c[0] - hx, c[1] - hy, c[2] + hz],
            [c[0] + hx, c[1] - hy, c[2] + hz],
            [c[0] + hx, c[1] + hy, c[2] + hz],
            [c[0] - hx, c[1] + hy, c[2] + hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return HullMesh.from_arrays(verts, faces)


def _plane_intersect(a: np.ndarray, b: np.ndarray, da: float, db: float) -> np.ndarray:
    denom = float(da - db)
    t = 0.5 if abs(denom) < 1e-18 else float(da / denom)
    t = min(1.0, max(0.0, t))
    return a + t * (b - a)


def _clip_triangle(tri: np.ndarray, depths: np.ndarray, eps: float = 1e-10):
    """Keep the underwater part of one triangle. depths > 0 is underwater."""
    under = depths >= -eps
    n_under = int(np.count_nonzero(under))
    if n_under == 0:
        return [], []
    if n_under == 3:
        segs = []
        for i in range(3):
            j = (i + 1) % 3
            if abs(float(depths[i])) <= eps and abs(float(depths[j])) <= eps:
                segs.append((tri[i].copy(), tri[j].copy()))
        return [tri.copy()], segs

    poly: list[np.ndarray] = []
    poly_d: list[float] = []
    for i in range(3):
        j = (i + 1) % 3
        a, b = tri[i], tri[j]
        da, db = float(depths[i]), float(depths[j])
        a_in, b_in = da >= -eps, db >= -eps
        if a_in and b_in:
            poly.append(b)
            poly_d.append(db)
        elif a_in and not b_in:
            p = _plane_intersect(a, b, da, db)
            poly.append(p)
            poly_d.append(0.0)
        elif (not a_in) and b_in:
            p = _plane_intersect(a, b, da, db)
            poly.append(p)
            poly_d.append(0.0)
            poly.append(b)
            poly_d.append(db)

    tris = []
    if len(poly) >= 3:
        origin = poly[0]
        for k in range(1, len(poly) - 1):
            tris.append(np.stack([origin, poly[k], poly[k + 1]], axis=0))

    segs = []
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        if abs(poly_d[i]) <= eps and abs(poly_d[j]) <= eps:
            segs.append((poly[i].copy(), poly[j].copy()))
    return tris, segs


def _accumulate_tris(tris: list[np.ndarray]) -> tuple[float, np.ndarray, float, np.ndarray, np.ndarray]:
    z3 = np.zeros(3, dtype=np.float64)
    if not tris:
        return 0.0, z3.copy(), 0.0, z3.copy(), z3.copy()
    arr = np.stack(tris, axis=0)
    v0, v1, v2 = arr[:, 0], arr[:, 1], arr[:, 2]
    cross = np.cross(v1, v2)
    vol6 = np.einsum("ij,ij->i", v0, cross)
    volume = float(vol6.sum() / 6.0)
    cob_acc = ((v0 + v1 + v2) * vol6[:, None]).sum(axis=0) / 24.0
    n = np.cross(v1 - v0, v2 - v0)
    S = 0.5 * n
    area = 0.5 * float(np.linalg.norm(n, axis=1).sum())
    area_pos = np.maximum(S, 0.0).sum(axis=0)
    area_neg = np.maximum(-S, 0.0).sum(axis=0)
    return volume, cob_acc, area, area_pos, area_neg


def _cap_from_segments(segs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    if len(segs) < 3:
        return []
    pts = np.concatenate([np.stack([a, b], axis=0) for a, b in segs], axis=0)
    ref = pts.mean(axis=0)
    return [np.stack([ref, a, b], axis=0) for a, b in segs]


def _finish_submerged(
    hull_v: float,
    hull_cob_acc: np.ndarray,
    area_pos: np.ndarray,
    area_neg: np.ndarray,
    segs: list[tuple[np.ndarray, np.ndarray]],
    mean_eta: float,
    sub_h: float,
    u_water: np.ndarray,
    a_water: np.ndarray,
    fallback_cob: np.ndarray,
) -> SubmergedInfo:
    cap_tris = _cap_from_segments(segs)
    cap_v, cap_cob_acc, wp_area, _, _ = _accumulate_tris(cap_tris)
    volume = float(hull_v + cap_v)
    hull_cob_acc = np.asarray(hull_cob_acc, dtype=np.float64).reshape(3)
    cap_cob_acc = np.asarray(cap_cob_acc, dtype=np.float64).reshape(3)
    if volume < 0.0:
        volume = -volume
        hull_cob_acc = -hull_cob_acc
        cap_cob_acc = -cap_cob_acc
    if volume <= 1e-18:
        return SubmergedInfo(
            volume=0.0,
            cob=np.asarray(fallback_cob, dtype=np.float64).reshape(3),
            waterplane_area=0.0,
            water_z=mean_eta,
            sub_h=0.0,
            u_water=np.zeros(3, dtype=np.float64),
            a_water=np.zeros(3, dtype=np.float64),
        )
    cob = (hull_cob_acc + cap_cob_acc) / volume
    return SubmergedInfo(
        volume=volume,
        cob=cob,
        waterplane_area=float(wp_area),
        water_z=mean_eta,
        sub_h=float(sub_h),
        area_pos=np.asarray(area_pos, dtype=np.float64).reshape(3),
        area_neg=np.asarray(area_neg, dtype=np.float64).reshape(3),
        u_water=np.asarray(u_water, dtype=np.float64).reshape(3),
        a_water=np.asarray(a_water, dtype=np.float64).reshape(3),
    )


def submerged_mesh(
    verts_world: np.ndarray,
    faces: np.ndarray,
    water_z: np.ndarray,
    vel=None,
    acc=None,
) -> SubmergedInfo:
    """Clip a closed mesh against z = water_z per vertex."""
    verts = np.asarray(verts_world, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    eta = np.asarray(water_z, dtype=np.float64).reshape(-1)
    depths = eta - verts[:, 2]
    mean_eta = float(eta.mean()) if eta.size else 0.0
    fallback = verts.mean(axis=0) if verts.size else np.zeros(3, dtype=np.float64)
    z3 = np.zeros(3, dtype=np.float64)
    vel_arr = z3 if vel is None else np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    acc_arr = z3 if acc is None else np.asarray(acc, dtype=np.float64).reshape(-1, 3)

    if faces.shape[0] == 0:
        return SubmergedInfo(0.0, fallback, 0.0, mean_eta, 0.0)

    face_d = depths[faces]
    n_under = np.count_nonzero(face_d >= -1e-10, axis=1)
    all_under = n_under == 3
    mixed = (n_under > 0) & (n_under < 3)

    tris: list[np.ndarray] = []
    segs: list[tuple[np.ndarray, np.ndarray]] = []

    if np.any(all_under):
        fu = faces[all_under]
        tris.extend(list(verts[fu]))
        du = face_d[all_under]
        for drow, idx in zip(du, fu):
            for i in range(3):
                j = (i + 1) % 3
                if abs(float(drow[i])) <= 1e-10 and abs(float(drow[j])) <= 1e-10:
                    segs.append((verts[idx[i]].copy(), verts[idx[j]].copy()))

    if np.any(mixed):
        fm = faces[mixed]
        for idx, drow in zip(fm, face_d[mixed]):
            clip_tris, clip_segs = _clip_triangle(verts[idx], drow)
            tris.extend(clip_tris)
            segs.extend(clip_segs)

    hull_v, hull_cob_acc, _hull_area, area_pos, area_neg = _accumulate_tris(tris)
    under = depths >= -1e-10
    if np.any(under):
        sub_h = float(np.mean(depths[under]))
        u_water = vel_arr[under].mean(axis=0) if vel_arr.ndim == 2 and vel_arr.shape[0] == verts.shape[0] else z3.copy()
        a_water = acc_arr[under].mean(axis=0) if acc_arr.ndim == 2 and acc_arr.shape[0] == verts.shape[0] else z3.copy()
    else:
        sub_h = 0.0
        u_water = z3.copy()
        a_water = z3.copy()
    return _finish_submerged(
        hull_v,
        hull_cob_acc,
        area_pos,
        area_neg,
        segs,
        mean_eta,
        sub_h,
        u_water,
        a_water,
        fallback,
    )


@dataclass
class HullMesh:
    verts: np.ndarray
    faces: np.ndarray
    volume: float
    cob_local: np.ndarray
    watertight: bool
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    _gpu: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_arrays(cls, verts, faces) -> "HullMesh":
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3).copy()
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3).copy()
        vol, cob = signed_volume_and_centroid(verts, faces)
        if vol < 0.0:
            faces = faces[:, [0, 2, 1]]
            vol, cob = signed_volume_and_centroid(verts, faces)
        watertight = edge_watertight(faces)
        mn = verts.min(axis=0)
        mx = verts.max(axis=0)
        return cls(
            verts=verts,
            faces=faces,
            volume=abs(float(vol)),
            cob_local=np.asarray(cob, dtype=np.float64).reshape(3),
            watertight=bool(watertight),
            aabb_min=mn,
            aabb_max=mx,
        )

    @classmethod
    def from_prims(cls, mesh_prim, body_prim) -> "HullMesh":
        from pxr import Usd, UsdGeom

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            raise RuntimeError(f"mesh has no geometry: {mesh_prim.GetPath()}")
        verts = np.array(points, dtype=np.float64).reshape(-1, 3)
        faces = triangulate_faces(counts, indices)
        m_mesh = np.array(
            UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        ).reshape(4, 4)
        m_body = np.array(
            UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        ).reshape(4, 4)
        m_to_body = np.linalg.inv(m_body) @ m_mesh
        homog = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
        local = (m_to_body @ homog.T).T[:, :3]
        return cls.from_arrays(local, faces)

    @property
    def n_verts(self) -> int:
        return int(self.verts.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def half_extents(self) -> np.ndarray:
        return 0.5 * (self.aabb_max - self.aabb_min)

    @property
    def aabb_center(self) -> np.ndarray:
        return 0.5 * (self.aabb_min + self.aabb_max)

    @property
    def xy_area(self) -> float:
        d = self.aabb_max - self.aabb_min
        return float(max(d[0] * d[1], 1e-9))

    def world_verts(self, pos, quat_wxyz) -> np.ndarray:
        R = quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        return self.verts @ R.T + p

    def world_aabb(self, pos, quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
        w = self.world_verts(pos, quat_wxyz)
        return w.min(axis=0), w.max(axis=0)

    def _ensure_gpu(self):
        if self._gpu is not None:
            return self._gpu
        try:
            from .warp_hull import WarpHull
            from .warp_gerstner import _warp

            if _warp() is None:
                return None
            self._gpu = WarpHull(self.verts, self.faces)
            return self._gpu
        except Exception:
            self._gpu = None
            return None

    def submerged(self, pos, quat_wxyz, water, prefer_gpu: bool = True) -> SubmergedInfo:
        position = np.asarray(pos, dtype=np.float64).reshape(3)
        # Keep Z in the water/world frame so a flat z=0 cap retains its exact
        # volume, but remove the large stage XY offset from moment integrals.
        origin = np.array([position[0], position[1], 0.0], dtype=np.float64)
        if prefer_gpu:
            gpu = self._ensure_gpu()
            if gpu is not None:
                raw = gpu.run(pos, quat_wxyz, water)
                if raw is not None:
                    segs = list(zip(raw.segs_a, raw.segs_b))
                    rotation = quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
                    fallback = rotation @ self.aabb_center + position - origin
                    info = _finish_submerged(
                        raw.hull_volume,
                        raw.hull_cob_acc,
                        raw.area_pos,
                        raw.area_neg,
                        segs,
                        raw.mean_eta,
                        raw.sub_h,
                        raw.u_water,
                        raw.a_water,
                        fallback,
                    )
                    # Warp integrates in a body-centred XY frame.  A wavy clip
                    # cap is only approximately closed, so stage-origin
                    # tetrahedra make errors grow with the boat's XY offset.
                    info.cob = info.cob + origin
                    return info
        world = self.world_verts(pos, quat_wxyz)
        if hasattr(water, "sample_kinematics"):
            eta, vel, acc = water.sample_kinematics(world[:, :2])
        else:
            eta = np.array([float(water.sample_height(p[0], p[1])) for p in world], dtype=np.float64)
            vel = acc = None
        info = submerged_mesh(
            world - origin,
            self.faces,
            eta,
            vel=vel,
            acc=acc,
        )
        info.cob = info.cob + origin
        return info
