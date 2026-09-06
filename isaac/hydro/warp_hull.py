"""GPU clip of a closed triangle hull against Gerstner water. Not a box."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .volume import quat_wxyz_to_rot
from .warp_gerstner import _warp, pack_trains, sample_kernel

_clip_kernel = None
_xform_kernel = None
_vert_kernel = None
_zero_f = None
_zero_i = None


def _kernels(wp):
    global _clip_kernel, _xform_kernel, _vert_kernel, _zero_f, _zero_i
    if _clip_kernel is not None:
        return

    @wp.func
    def _add_tri(
        v0: wp.vec3,
        v1: wp.vec3,
        v2: wp.vec3,
        vol: wp.array(dtype=float),
        cob: wp.array(dtype=float),
        apos: wp.array(dtype=float),
        aneg: wp.array(dtype=float),
        with_proj: int,
    ):
        cr = wp.cross(v1, v2)
        vol6 = wp.dot(v0, cr)
        wp.atomic_add(vol, 0, vol6 / 6.0)
        s = (v0 + v1 + v2) * (vol6 / 24.0)
        wp.atomic_add(cob, 0, s[0])
        wp.atomic_add(cob, 1, s[1])
        wp.atomic_add(cob, 2, s[2])
        if with_proj == 1:
            n = wp.cross(v1 - v0, v2 - v0)
            hx = n[0] * 0.5
            hy = n[1] * 0.5
            hz = n[2] * 0.5
            wp.atomic_add(apos, 0, wp.max(hx, 0.0))
            wp.atomic_add(apos, 1, wp.max(hy, 0.0))
            wp.atomic_add(apos, 2, wp.max(hz, 0.0))
            wp.atomic_add(aneg, 0, wp.max(-hx, 0.0))
            wp.atomic_add(aneg, 1, wp.max(-hy, 0.0))
            wp.atomic_add(aneg, 2, wp.max(-hz, 0.0))

    @wp.func
    def _intersect(a: wp.vec3, b: wp.vec3, da: float, db: float) -> wp.vec3:
        denom = da - db
        t = float(0.5)
        if wp.abs(denom) > 1.0e-18:
            t = da / denom
        t = wp.clamp(t, 0.0, 1.0)
        return a + t * (b - a)

    @wp.func
    def _push_seg(
        a: wp.vec3,
        b: wp.vec3,
        seg_a: wp.array(dtype=wp.vec3),
        seg_b: wp.array(dtype=wp.vec3),
        n_seg: wp.array(dtype=int),
        max_seg: int,
    ):
        idx = wp.atomic_add(n_seg, 0, 1)
        if idx < max_seg:
            seg_a[idx] = a
            seg_b[idx] = b

    @wp.kernel
    def xform_verts(local: wp.array(dtype=wp.vec3), world: wp.array(dtype=wp.vec3), R: wp.mat33, p: wp.vec3):
        i = wp.tid()
        world[i] = R * local[i] + p

    @wp.kernel
    def vert_stats(
        world: wp.array(dtype=wp.vec3),
        eta: wp.array(dtype=float),
        vel: wp.array(dtype=wp.vec3),
        acc: wp.array(dtype=wp.vec3),
        u_sum: wp.array(dtype=float),
        a_sum: wp.array(dtype=float),
        n_under: wp.array(dtype=int),
        eta_sum: wp.array(dtype=float),
        depth_sum: wp.array(dtype=float),
        n_verts: wp.array(dtype=int),
    ):
        i = wp.tid()
        e = eta[i]
        z = world[i][2]
        d = e - z
        wp.atomic_add(eta_sum, 0, e)
        wp.atomic_add(n_verts, 0, 1)
        if d >= -1.0e-10:
            v = vel[i]
            ac = acc[i]
            wp.atomic_add(n_under, 0, 1)
            wp.atomic_add(depth_sum, 0, d)
            wp.atomic_add(u_sum, 0, v[0])
            wp.atomic_add(u_sum, 1, v[1])
            wp.atomic_add(u_sum, 2, v[2])
            wp.atomic_add(a_sum, 0, ac[0])
            wp.atomic_add(a_sum, 1, ac[1])
            wp.atomic_add(a_sum, 2, ac[2])

    @wp.kernel
    def clip_faces(
        world: wp.array(dtype=wp.vec3),
        faces: wp.array(dtype=wp.vec3i),
        eta: wp.array(dtype=float),
        origin: wp.vec3,
        vol: wp.array(dtype=float),
        cob: wp.array(dtype=float),
        apos: wp.array(dtype=float),
        aneg: wp.array(dtype=float),
        seg_a: wp.array(dtype=wp.vec3),
        seg_b: wp.array(dtype=wp.vec3),
        n_seg: wp.array(dtype=int),
        max_seg: int,
    ):
        fi = wp.tid()
        idx = faces[fi]
        i0 = idx[0]
        i1 = idx[1]
        i2 = idx[2]
        # Accumulate clipped tetrahedra around the rigid body's XY origin.
        # Keep world Z so a flat z=0 cap retains exact volume.  The sampled
        # wave cap is only approximately closed; stage XY magnifies its error.
        p0 = world[i0] - origin
        p1 = world[i1] - origin
        p2 = world[i2] - origin
        d0 = (eta[i0] - origin[2]) - p0[2]
        d1 = (eta[i1] - origin[2]) - p1[2]
        d2 = (eta[i2] - origin[2]) - p2[2]
        eps = float(1.0e-10)
        n_under = int(0)
        if d0 >= -eps:
            n_under = n_under + 1
        if d1 >= -eps:
            n_under = n_under + 1
        if d2 >= -eps:
            n_under = n_under + 1
        if n_under == 0:
            return
        if n_under == 3:
            _add_tri(p0, p1, p2, vol, cob, apos, aneg, 1)
            if wp.abs(d0) <= eps and wp.abs(d1) <= eps:
                _push_seg(p0, p1, seg_a, seg_b, n_seg, max_seg)
            if wp.abs(d1) <= eps and wp.abs(d2) <= eps:
                _push_seg(p1, p2, seg_a, seg_b, n_seg, max_seg)
            if wp.abs(d2) <= eps and wp.abs(d0) <= eps:
                _push_seg(p2, p0, seg_a, seg_b, n_seg, max_seg)
            return

        q0 = p0
        q1 = p0
        q2 = p0
        q3 = p0
        e0 = float(0.0)
        e1 = float(0.0)
        e2 = float(0.0)
        e3 = float(0.0)
        npoly = int(0)

        for edge in range(3):
            a = p0
            b = p1
            da = d0
            db = d1
            if edge == 1:
                a = p1
                b = p2
                da = d1
                db = d2
            elif edge == 2:
                a = p2
                b = p0
                da = d2
                db = d0
            a_in = da >= -eps
            b_in = db >= -eps
            if a_in and b_in:
                if npoly == 0:
                    q0 = b
                    e0 = db
                elif npoly == 1:
                    q1 = b
                    e1 = db
                elif npoly == 2:
                    q2 = b
                    e2 = db
                elif npoly == 3:
                    q3 = b
                    e3 = db
                npoly = npoly + 1
            elif a_in and not b_in:
                ip = _intersect(a, b, da, db)
                if npoly == 0:
                    q0 = ip
                    e0 = float(0.0)
                elif npoly == 1:
                    q1 = ip
                    e1 = float(0.0)
                elif npoly == 2:
                    q2 = ip
                    e2 = float(0.0)
                elif npoly == 3:
                    q3 = ip
                    e3 = float(0.0)
                npoly = npoly + 1
            elif (not a_in) and b_in:
                ip = _intersect(a, b, da, db)
                if npoly == 0:
                    q0 = ip
                    e0 = float(0.0)
                elif npoly == 1:
                    q1 = ip
                    e1 = float(0.0)
                elif npoly == 2:
                    q2 = ip
                    e2 = float(0.0)
                elif npoly == 3:
                    q3 = ip
                    e3 = float(0.0)
                npoly = npoly + 1
                if npoly == 1:
                    q1 = b
                    e1 = db
                elif npoly == 2:
                    q2 = b
                    e2 = db
                elif npoly == 3:
                    q3 = b
                    e3 = db
                npoly = npoly + 1

        if npoly >= 3:
            _add_tri(q0, q1, q2, vol, cob, apos, aneg, 1)
        if npoly >= 4:
            _add_tri(q0, q2, q3, vol, cob, apos, aneg, 1)

        if npoly >= 2:
            if wp.abs(e0) <= eps and wp.abs(e1) <= eps:
                _push_seg(q0, q1, seg_a, seg_b, n_seg, max_seg)
            if npoly >= 3 and wp.abs(e1) <= eps and wp.abs(e2) <= eps:
                _push_seg(q1, q2, seg_a, seg_b, n_seg, max_seg)
            if npoly >= 4 and wp.abs(e2) <= eps and wp.abs(e3) <= eps:
                _push_seg(q2, q3, seg_a, seg_b, n_seg, max_seg)
            last_e = e2
            last_q = q2
            if npoly >= 4:
                last_e = e3
                last_q = q3
            if npoly >= 3 and wp.abs(last_e) <= eps and wp.abs(e0) <= eps:
                _push_seg(last_q, q0, seg_a, seg_b, n_seg, max_seg)

    @wp.kernel
    def zero_float(a: wp.array(dtype=float)):
        a[wp.tid()] = float(0.0)

    @wp.kernel
    def zero_int(a: wp.array(dtype=int)):
        a[wp.tid()] = int(0)

    _clip_kernel = clip_faces
    _xform_kernel = xform_verts
    _vert_kernel = vert_stats
    _zero_f = zero_float
    _zero_i = zero_int


@dataclass
class HullClipRaw:
    hull_volume: float
    hull_cob_acc: np.ndarray
    area_pos: np.ndarray
    area_neg: np.ndarray
    segs_a: np.ndarray
    segs_b: np.ndarray
    u_water: np.ndarray
    a_water: np.ndarray
    mean_eta: float
    sub_h: float


class WarpHull:
    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        wp = _warp()
        if wp is None:
            raise RuntimeError("warp not available")
        _kernels(wp)
        self.device = wp.get_device()
        verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.n = int(verts.shape[0])
        self.n_faces = int(faces.shape[0])
        self.max_seg = max(self.n_faces * 2, 8)
        self.local = wp.array(verts, dtype=wp.vec3, device=self.device)
        self.faces = wp.array(faces, dtype=wp.vec3i, device=self.device)
        self.world = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.eta = wp.zeros(self.n, dtype=float, device=self.device)
        self.vel = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.acc = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.seg_a = wp.zeros(self.max_seg, dtype=wp.vec3, device=self.device)
        self.seg_b = wp.zeros(self.max_seg, dtype=wp.vec3, device=self.device)
        self.n_seg = wp.zeros(1, dtype=int, device=self.device)
        self.vol = wp.zeros(1, dtype=float, device=self.device)
        self.cob = wp.zeros(3, dtype=float, device=self.device)
        self.apos = wp.zeros(3, dtype=float, device=self.device)
        self.aneg = wp.zeros(3, dtype=float, device=self.device)
        self.u_sum = wp.zeros(3, dtype=float, device=self.device)
        self.a_sum = wp.zeros(3, dtype=float, device=self.device)
        self.n_under = wp.zeros(1, dtype=int, device=self.device)
        self.eta_sum = wp.zeros(1, dtype=float, device=self.device)
        self.depth_sum = wp.zeros(1, dtype=float, device=self.device)
        self.n_verts = wp.zeros(1, dtype=int, device=self.device)
        self._wave_key = None
        self.n_waves = 0
        self.rest_z = 0.0
        self.amp = wp.zeros(1, dtype=float, device=self.device)
        self.k = wp.zeros(1, dtype=float, device=self.device)
        self.omega = wp.zeros(1, dtype=float, device=self.device)
        self.dirx = wp.zeros(1, dtype=float, device=self.device)
        self.diry = wp.zeros(1, dtype=float, device=self.device)
        self.steep = wp.zeros(1, dtype=float, device=self.device)
        self.phase = wp.zeros(1, dtype=float, device=self.device)

    def _bind_water(self, water) -> None:
        wp = _warp()
        field = getattr(water, "waves", None)
        key = (id(field) if field is not None else None, float(getattr(water, "water_z", 0.0)))
        if key == self._wave_key:
            return
        self._wave_key = key
        if field is None:
            n_waves, arrays, _rest = pack_trains(None)
            self.n_waves = 0
            self.rest_z = float(getattr(water, "water_z", 0.0))
        else:
            n_waves, arrays, rest_z = pack_trains(field)
            self.n_waves = int(n_waves)
            self.rest_z = float(rest_z)
        self.amp = wp.array(arrays["amp"], device=self.device)
        self.k = wp.array(arrays["k"], device=self.device)
        self.omega = wp.array(arrays["omega"], device=self.device)
        self.dirx = wp.array(arrays["dirx"], device=self.device)
        self.diry = wp.array(arrays["diry"], device=self.device)
        self.steep = wp.array(arrays["steep"], device=self.device)
        self.phase = wp.array(arrays["phase"], device=self.device)

    def _zero(self) -> None:
        wp = _warp()
        wp.launch(_zero_f, dim=1, inputs=[self.vol], device=self.device)
        wp.launch(_zero_f, dim=3, inputs=[self.cob], device=self.device)
        wp.launch(_zero_f, dim=3, inputs=[self.apos], device=self.device)
        wp.launch(_zero_f, dim=3, inputs=[self.aneg], device=self.device)
        wp.launch(_zero_f, dim=3, inputs=[self.u_sum], device=self.device)
        wp.launch(_zero_f, dim=3, inputs=[self.a_sum], device=self.device)
        wp.launch(_zero_f, dim=1, inputs=[self.eta_sum], device=self.device)
        wp.launch(_zero_f, dim=1, inputs=[self.depth_sum], device=self.device)
        wp.launch(_zero_i, dim=1, inputs=[self.n_seg], device=self.device)
        wp.launch(_zero_i, dim=1, inputs=[self.n_under], device=self.device)
        wp.launch(_zero_i, dim=1, inputs=[self.n_verts], device=self.device)

    def run(self, pos, quat_wxyz, water) -> HullClipRaw | None:
        wp = _warp()
        if wp is None or _clip_kernel is None:
            return None
        self._bind_water(water)
        self._zero()
        R = quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64).reshape(4)).astype(np.float32)
        p = np.asarray(pos, dtype=np.float32).reshape(3)
        R_wp = wp.mat33(
            float(R[0, 0]), float(R[0, 1]), float(R[0, 2]),
            float(R[1, 0]), float(R[1, 1]), float(R[1, 2]),
            float(R[2, 0]), float(R[2, 1]), float(R[2, 2]),
        )
        p_wp = wp.vec3(float(p[0]), float(p[1]), float(p[2]))
        clip_origin_wp = wp.vec3(float(p[0]), float(p[1]), 0.0)
        wp.launch(_xform_kernel, dim=self.n, inputs=[self.local, self.world, R_wp, p_wp], device=self.device)
        t = float(getattr(water, "time", 0.0))
        samp = sample_kernel()
        if samp is None:
            return None
        wp.launch(
            samp,
            dim=self.n,
            inputs=[
                self.world,
                self.eta,
                self.vel,
                self.acc,
                t,
                float(self.rest_z),
                int(self.n_waves),
                self.amp,
                self.k,
                self.omega,
                self.dirx,
                self.diry,
                self.steep,
                self.phase,
            ],
            device=self.device,
        )
        field = getattr(water, "waves", None)
        fft = getattr(field, "fft", None) if field is not None else None
        if fft is not None:
            fft.evolve(t)
            world_np = self.world.numpy()
            h, _dx, _dy, _sx, _sy, vz = fft.sample(world_np[:, :2])
            eta = self.eta.numpy()
            vel = self.vel.numpy()
            eta = eta + h.astype(np.float32)
            vel[:, 2] = vel[:, 2] + vz.astype(np.float32)
            self.eta.assign(eta)
            self.vel.assign(vel)
        wp.launch(
            _vert_kernel,
            dim=self.n,
            inputs=[
                self.world,
                self.eta,
                self.vel,
                self.acc,
                self.u_sum,
                self.a_sum,
                self.n_under,
                self.eta_sum,
                self.depth_sum,
                self.n_verts,
            ],
            device=self.device,
        )
        if self.n_faces > 0:
            wp.launch(
                _clip_kernel,
                dim=self.n_faces,
                inputs=[
                    self.world,
                    self.faces,
                    self.eta,
                    clip_origin_wp,
                    self.vol,
                    self.cob,
                    self.apos,
                    self.aneg,
                    self.seg_a,
                    self.seg_b,
                    self.n_seg,
                    int(self.max_seg),
                ],
                device=self.device,
            )
        n_seg = int(self.n_seg.numpy()[0])
        n_seg = max(0, min(n_seg, self.max_seg))
        n_under = int(self.n_under.numpy()[0])
        n_verts = max(int(self.n_verts.numpy()[0]), 1)
        u = self.u_sum.numpy()
        a = self.a_sum.numpy()
        if n_under > 0:
            u = u / float(n_under)
            a = a / float(n_under)
        else:
            u = np.zeros(3, dtype=np.float64)
            a = np.zeros(3, dtype=np.float64)
        mean_eta = float(self.eta_sum.numpy()[0]) / float(n_verts)
        sub_h = float(self.depth_sum.numpy()[0]) / float(n_under) if n_under > 0 else 0.0
        segs_a = self.seg_a.numpy()[:n_seg] if n_seg else np.zeros((0, 3), dtype=np.float32)
        segs_b = self.seg_b.numpy()[:n_seg] if n_seg else np.zeros((0, 3), dtype=np.float32)
        return HullClipRaw(
            hull_volume=float(self.vol.numpy()[0]),
            hull_cob_acc=np.asarray(self.cob.numpy(), dtype=np.float64).reshape(3),
            area_pos=np.asarray(self.apos.numpy(), dtype=np.float64).reshape(3),
            area_neg=np.asarray(self.aneg.numpy(), dtype=np.float64).reshape(3),
            segs_a=np.asarray(segs_a, dtype=np.float64).reshape(-1, 3),
            segs_b=np.asarray(segs_b, dtype=np.float64).reshape(-1, 3),
            u_water=np.asarray(u, dtype=np.float64).reshape(3),
            a_water=np.asarray(a, dtype=np.float64).reshape(3),
            mean_eta=mean_eta,
            sub_h=sub_h,
        )
