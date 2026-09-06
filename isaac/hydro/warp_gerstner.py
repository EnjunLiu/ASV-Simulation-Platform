"""Gerstner displacement + analytic normals on NVIDIA Warp. Optional FFT / wake overlay."""

from __future__ import annotations

import numpy as np

_wp = None
_kernel = None
_sample = None
_detail = None
_failed = False


def pack_trains(field, min_wavelength: float = 0.0) -> tuple[int, dict[str, np.ndarray], float]:
    """Return (n_waves, numpy arrays, rest_z). Arrays have length >= 1."""
    amp, k, omega, dirx, diry, steep, phase = [], [], [], [], [], [], []
    g = float(getattr(field, "gravity", 9.81)) if field is not None else 9.81
    rest_z = float(getattr(field, "rest_z", 0.0)) if field is not None else 0.0
    trains = list(getattr(field, "trains", None) or [])
    min_lam = float(min_wavelength)
    for w in trains:
        if float(w.wavelength) + 1e-9 < min_lam:
            continue
        d = np.array([float(w.direction[0]), float(w.direction[1])], dtype=np.float64)
        nrm = float(np.linalg.norm(d))
        d = np.array([1.0, 0.0]) if nrm < 1e-12 else d / nrm
        kk = 2.0 * np.pi / max(float(w.wavelength), 1e-6)
        amp.append(float(w.amplitude))
        k.append(kk)
        omega.append(float(np.sqrt(max(g * kk, 0.0))))
        dirx.append(float(d[0]))
        diry.append(float(d[1]))
        steep.append(float(np.clip(w.steepness, 0.0, 1.0)))
        phase.append(float(w.phase))
    n_waves = len(amp)
    if n_waves == 0:
        amp, k, omega, dirx, diry, steep, phase = [0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0]
    arrays = {
        "amp": np.array(amp, dtype=np.float32),
        "k": np.array(k, dtype=np.float32),
        "omega": np.array(omega, dtype=np.float32),
        "dirx": np.array(dirx, dtype=np.float32),
        "diry": np.array(diry, dtype=np.float32),
        "steep": np.array(steep, dtype=np.float32),
        "phase": np.array(phase, dtype=np.float32),
    }
    return n_waves, arrays, rest_z


def _warp():
    global _wp, _kernel, _sample, _detail, _failed
    if _failed:
        return None
    if _wp is not None:
        return _wp
    try:
        import warp as wp

        wp.init()
        _kernel, _sample, _detail = _define_kernels(wp)
        _wp = wp
        print("Warp Gerstner device:", wp.get_device())
        return wp
    except Exception as exc:
        print("Warp Gerstner unavailable, CPU numpy path:", exc)
        _failed = True
        return None


def sample_kernel():
    _warp()
    return _sample


def _define_kernels(wp):
    @wp.func
    def _fract(x: float) -> float:
        return x - wp.floor(x)

    @wp.func
    def _bilerp(arr: wp.array(dtype=float), n: int, u: float, v: float) -> float:
        uf = _fract(u) * float(n)
        vf = _fract(v) * float(n)
        i0 = int(wp.floor(uf))
        j0 = int(wp.floor(vf))
        su = uf - wp.floor(uf)
        sv = vf - wp.floor(vf)
        npos = n if n > 0 else 1
        i0 = i0 % npos
        j0 = j0 % npos
        if i0 < 0:
            i0 = i0 + npos
        if j0 < 0:
            j0 = j0 + npos
        i1 = (i0 + 1) % npos
        j1 = (j0 + 1) % npos
        a00 = arr[j0 * npos + i0]
        a10 = arr[j0 * npos + i1]
        a01 = arr[j1 * npos + i0]
        a11 = arr[j1 * npos + i1]
        return (a00 * (1.0 - su) + a10 * su) * (1.0 - sv) + (a01 * (1.0 - su) + a11 * su) * sv

    @wp.kernel
    def gerstner_deform(
        rest: wp.array(dtype=wp.vec3),
        out_pts: wp.array(dtype=wp.vec3),
        out_nrm: wp.array(dtype=wp.vec3),
        t: float,
        rest_z: float,
        origin_x: float,
        origin_y: float,
        n_waves: int,
        amp: wp.array(dtype=float),
        k: wp.array(dtype=float),
        omega: wp.array(dtype=float),
        dirx: wp.array(dtype=float),
        diry: wp.array(dtype=float),
        steep: wp.array(dtype=float),
        phase: wp.array(dtype=float),
    ):
        i = wp.tid()
        p = rest[i]
        x = p[0] + origin_x
        y = p[1] + origin_y
        dx = float(0.0)
        dy = float(0.0)
        z = rest_z
        nx = float(0.0)
        ny = float(0.0)
        nz = float(1.0)
        for w in range(n_waves):
            a = amp[w]
            kk = k[w]
            qi = steep[w] * a
            th = kk * (dirx[w] * x + diry[w] * y) - omega[w] * t + phase[w]
            cth = wp.cos(th)
            sth = wp.sin(th)
            dx = dx + dirx[w] * qi * cth
            dy = dy + diry[w] * qi * cth
            z = z + a * sth
            nx = nx - dirx[w] * a * kk * cth
            ny = ny - diry[w] * a * kk * cth
            nz = nz - qi * kk * sth
        inv = float(1.0) / wp.sqrt(nx * nx + ny * ny + nz * nz + 1.0e-12)
        out_pts[i] = wp.vec3(p[0] + dx, p[1] + dy, z)
        out_nrm[i] = wp.vec3(nx * inv, ny * inv, nz * inv)

    @wp.kernel
    def gerstner_sample(
        pts: wp.array(dtype=wp.vec3),
        out_eta: wp.array(dtype=float),
        out_vel: wp.array(dtype=wp.vec3),
        out_acc: wp.array(dtype=wp.vec3),
        t: float,
        rest_z: float,
        n_waves: int,
        amp: wp.array(dtype=float),
        k: wp.array(dtype=float),
        omega: wp.array(dtype=float),
        dirx: wp.array(dtype=float),
        diry: wp.array(dtype=float),
        steep: wp.array(dtype=float),
        phase: wp.array(dtype=float),
    ):
        i = wp.tid()
        p = pts[i]
        x = p[0]
        y = p[1]
        eta = rest_z
        vx = float(0.0)
        vy = float(0.0)
        vz = float(0.0)
        ax = float(0.0)
        ay = float(0.0)
        az = float(0.0)
        for w in range(n_waves):
            a = amp[w]
            kk = k[w]
            om = omega[w]
            qi = steep[w] * a
            th = kk * (dirx[w] * x + diry[w] * y) - om * t + phase[w]
            cth = wp.cos(th)
            sth = wp.sin(th)
            eta = eta + a * sth
            vx = vx + dirx[w] * qi * om * sth
            vy = vy + diry[w] * qi * om * sth
            vz = vz - a * om * cth
            ax = ax - dirx[w] * qi * om * om * cth
            ay = ay - diry[w] * qi * om * om * cth
            az = az - a * om * om * sth
        out_eta[i] = eta
        out_vel[i] = wp.vec3(vx, vy, vz)
        out_acc[i] = wp.vec3(ax, ay, az)

    @wp.kernel
    def apply_fft_wake(
        rest: wp.array(dtype=wp.vec3),
        out_pts: wp.array(dtype=wp.vec3),
        out_nrm: wp.array(dtype=wp.vec3),
        origin_x: float,
        origin_y: float,
        fft_n: int,
        fft_tile: float,
        fft_h: wp.array(dtype=float),
        fft_dx: wp.array(dtype=float),
        fft_dy: wp.array(dtype=float),
        fft_sx: wp.array(dtype=float),
        fft_sy: wp.array(dtype=float),
        fade_inner: float,
        fade_outer: float,
        wake_on: int,
        wake_px: float,
        wake_py: float,
        wake_hx: float,
        wake_hy: float,
        wake_speed: float,
        wake_len: float,
        wake_beam: float,
        wake_amp: float,
        wake_g: float,
    ):
        i = wp.tid()
        p = rest[i]
        q = out_pts[i]
        nrm = out_nrm[i]
        wx = p[0] + origin_x
        wy = p[1] + origin_y
        fade = float(1.0)
        r = wp.max(wp.abs(p[0]), wp.abs(p[1]))
        if fade_outer > fade_inner + 1.0e-4:
            fade = 1.0 - (r - fade_inner) / (fade_outer - fade_inner)
            fade = wp.clamp(fade, 0.0, 1.0)
        dh = float(0.0)
        ddx = float(0.0)
        ddy = float(0.0)
        dsx = float(0.0)
        dsy = float(0.0)
        if fft_n > 1 and fft_tile > 1.0e-6:
            u = wx / fft_tile
            v = wy / fft_tile
            dh = dh + _bilerp(fft_h, fft_n, u, v)
            ddx = ddx + _bilerp(fft_dx, fft_n, u, v)
            ddy = ddy + _bilerp(fft_dy, fft_n, u, v)
            dsx = dsx + _bilerp(fft_sx, fft_n, u, v)
            dsy = dsy + _bilerp(fft_sy, fft_n, u, v)
        if wake_on == 1 and wake_speed > 0.05 and wake_amp != 0.0:
            rx = (wx - wake_px) * wake_hx + (wy - wake_py) * wake_hy
            ry = -(wx - wake_px) * wake_hy + (wy - wake_py) * wake_hx
            aft = -rx
            k = wake_g / wp.max(wake_speed * wake_speed, 0.15)
            lam = 6.28318530718 / wp.max(k, 1.0e-3)
            amp = wake_amp * wp.min((wake_speed / 0.8) * (wake_speed / 0.8), 2.5)
            sigma = 0.18 * wake_len + 0.10 * wp.max(aft, 0.0)
            arm = wp.abs(ry) - 0.353553 * wp.max(aft, 0.0)
            env = wp.exp(-wp.max(aft, 0.0) / (9.0 * wake_len)) * wp.exp(-(arm * arm) / (sigma * sigma + 1.0e-6))
            env = env * wp.exp(-(ry * ry) / ((2.4 * wake_beam + 0.25 * wp.max(aft, 0.0)) * (2.4 * wake_beam + 0.25 * wp.max(aft, 0.0)) + 1.0e-6))
            trans = wp.sin(6.28318530718 * aft / lam)
            diverg = wp.sin(6.28318530718 * wp.sqrt(aft * aft + ry * ry) / (0.7 * lam))
            bow = wp.exp(-((rx - 0.35 * wake_len) * (rx - 0.35 * wake_len)) / ((0.18 * wake_len) * (0.18 * wake_len)))
            bow = bow * wp.exp(-(ry * ry) / ((0.35 * wake_beam) * (0.35 * wake_beam)))
            eta_w = amp * env * (0.65 * trans + 0.35 * diverg) + 0.35 * amp * bow
            if aft < -0.15 * wake_len and bow < 0.02:
                eta_w = float(0.0)
            if wp.sqrt(aft * aft + ry * ry) > 18.0 * wake_len:
                eta_w = float(0.0)
            dh = dh + eta_w
        dh = dh * fade
        ddx = ddx * fade
        ddy = ddy * fade
        dsx = dsx * fade
        dsy = dsy * fade
        out_pts[i] = wp.vec3(q[0] + ddx, q[1] + ddy, q[2] + dh)
        nx = nrm[0] - dsx
        ny = nrm[1] - dsy
        nz = nrm[2]
        inv = float(1.0) / wp.sqrt(nx * nx + ny * ny + nz * nz + 1.0e-12)
        out_nrm[i] = wp.vec3(nx * inv, ny * inv, nz * inv)

    return gerstner_deform, gerstner_sample, apply_fft_wake


class WarpGerstnerGrid:
    def __init__(self, field, rest_xyz: np.ndarray, min_wavelength: float = 0.0):
        wp = _warp()
        if wp is None:
            raise RuntimeError("warp not available")
        rest = np.asarray(rest_xyz, dtype=np.float32).reshape(-1, 3)
        self.n = int(rest.shape[0])
        self.device = wp.get_device()
        self.rest = wp.array(rest, dtype=wp.vec3, device=self.device)
        self.out = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.nrm = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        n_waves, arrays, rest_z = pack_trains(field, min_wavelength=min_wavelength)
        self.n_waves = int(n_waves)
        self.amp = wp.array(arrays["amp"], device=self.device)
        self.k = wp.array(arrays["k"], device=self.device)
        self.omega = wp.array(arrays["omega"], device=self.device)
        self.dirx = wp.array(arrays["dirx"], device=self.device)
        self.diry = wp.array(arrays["diry"], device=self.device)
        self.steep = wp.array(arrays["steep"], device=self.device)
        self.phase = wp.array(arrays["phase"], device=self.device)
        self.rest_z = float(rest_z)
        self._fft_n = 1
        self._fft_tile = 1.0
        self.fft_h = wp.zeros(1, dtype=float, device=self.device)
        self.fft_dx = wp.zeros(1, dtype=float, device=self.device)
        self.fft_dy = wp.zeros(1, dtype=float, device=self.device)
        self.fft_sx = wp.zeros(1, dtype=float, device=self.device)
        self.fft_sy = wp.zeros(1, dtype=float, device=self.device)

    def set_fft(self, field) -> None:
        wp = _wp
        fft = getattr(field, "fft", None) if field is not None else None
        if fft is None:
            self._fft_n = 1
            return
        n = int(fft.n)
        need = n * n
        if int(self.fft_h.shape[0]) != need:
            self.fft_h = wp.zeros(need, dtype=float, device=self.device)
            self.fft_dx = wp.zeros(need, dtype=float, device=self.device)
            self.fft_dy = wp.zeros(need, dtype=float, device=self.device)
            self.fft_sx = wp.zeros(need, dtype=float, device=self.device)
            self.fft_sy = wp.zeros(need, dtype=float, device=self.device)
        self.fft_h.assign(np.ascontiguousarray(fft.height, dtype=np.float32).reshape(-1))
        self.fft_dx.assign(np.ascontiguousarray(fft.dx, dtype=np.float32).reshape(-1))
        self.fft_dy.assign(np.ascontiguousarray(fft.dy, dtype=np.float32).reshape(-1))
        self.fft_sx.assign(np.ascontiguousarray(fft.sx, dtype=np.float32).reshape(-1))
        self.fft_sy.assign(np.ascontiguousarray(fft.sy, dtype=np.float32).reshape(-1))
        self._fft_n = n
        self._fft_tile = float(fft.tile_m)

    def launch(self, t: float, origin_xy=(0.0, 0.0), wake=None, fade_inner: float = 0.0, fade_outer: float = 0.0) -> None:
        wp = _wp
        ox = float(origin_xy[0])
        oy = float(origin_xy[1])
        wp.launch(
            _kernel,
            dim=self.n,
            inputs=[
                self.rest, self.out, self.nrm, float(t), self.rest_z, ox, oy, self.n_waves,
                self.amp, self.k, self.omega, self.dirx, self.diry, self.steep, self.phase,
            ],
            device=self.device,
        )
        fft_on = self._fft_n > 1
        wake_on = wake is not None and float(getattr(wake, "speed", 0.0)) > 0.05
        if not fft_on and not wake_on:
            return
        if _detail is None:
            return
        if wake is None:
            wpx = wpy = whx = 0.0
            why = 1.0
            wspd = wlen = wbeam = wamp = 0.0
            wg = 9.81
            wflag = 0
        else:
            wpx = float(wake.x)
            wpy = float(wake.y)
            n = float(np.hypot(float(wake.hx), float(wake.hy)))
            whx = float(wake.hx) / n if n > 1e-8 else 1.0
            why = float(wake.hy) / n if n > 1e-8 else 0.0
            wspd = float(wake.speed)
            wlen = max(float(wake.length), 0.2)
            wbeam = max(float(wake.beam), 0.1)
            wamp = float(wake.amp)
            wg = float(getattr(wake, "gravity", 9.81))
            wflag = 1
        wp.launch(
            _detail,
            dim=self.n,
            inputs=[
                self.rest, self.out, self.nrm, ox, oy,
                int(self._fft_n if fft_on else 1), float(self._fft_tile),
                self.fft_h, self.fft_dx, self.fft_dy, self.fft_sx, self.fft_sy,
                float(fade_inner), float(fade_outer),
                int(wflag), wpx, wpy, whx, why, wspd, wlen, wbeam, wamp, wg,
            ],
            device=self.device,
        )

    def deform(self, t: float) -> np.ndarray:
        self.launch(t)
        return self.out.numpy()
