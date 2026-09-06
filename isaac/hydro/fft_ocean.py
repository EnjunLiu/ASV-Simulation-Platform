"""Time-evolved Tessendorf / Phillips FFT. Displacement + slopes, not a static UV stamp."""

from __future__ import annotations

import numpy as np

G = 9.81


def _negk(arr: np.ndarray) -> np.ndarray:
    """a[-k] for numpy FFT-ordered 2D arrays."""
    return np.roll(np.roll(arr[::-1, ::-1], 1, axis=0), 1, axis=1)


class TessendorfField:
    """Periodic height / chop / slope maps. Call evolve(t) then sample or upload."""

    def __init__(
        self,
        n: int = 256,
        tile_m: float = 64.0,
        *,
        lam_min: float = 0.5,
        lam_max: float = 3.5,
        wind: tuple[float, float] = (1.0, 0.18),
        wind_speed: float = 7.0,
        height_rms: float = 0.03,
        choppiness: float = 0.85,
        seed: int = 7,
    ):
        n = int(n)
        if n < 8 or (n & (n - 1)):
            raise ValueError("FFT n must be a power of two, >= 8")
        self.n = n
        self.tile_m = float(tile_m)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.choppiness = float(choppiness)
        self.height_rms = float(height_rms)
        self._t = None
        dx = self.tile_m / float(n)
        freq = np.fft.fftfreq(n, d=dx) * (2.0 * np.pi)
        kx, ky = np.meshgrid(freq, freq, indexing="xy")
        k2 = kx * kx + ky * ky
        k = np.sqrt(k2)
        wd = np.array([float(wind[0]), float(wind[1])], dtype=np.float64)
        wn = float(np.linalg.norm(wd))
        wd = wd / wn if wn > 1e-12 else np.array([1.0, 0.0])
        khx = np.divide(kx, k, out=np.zeros_like(kx), where=k > 1e-12)
        khy = np.divide(ky, k, out=np.zeros_like(ky), where=k > 1e-12)
        mu = khx * wd[0] + khy * wd[1]
        spread = mu * mu + 0.18
        Lw = max(float(wind_speed) ** 2 / G, 1e-3)
        lam_min = max(float(lam_min), 2.2 * dx)
        lam_max = min(float(lam_max), 0.48 * self.tile_m)
        if lam_min >= lam_max:
            lam_min = 0.4 * lam_max
        self.lam_min = lam_min
        self.lam_max = lam_max
        k_lo = 2.0 * np.pi / lam_max
        k_hi = 2.0 * np.pi / lam_min
        l_small = 0.5 * lam_min
        valid = (k >= k_lo) & (k <= k_hi)
        P = np.zeros_like(k2)
        P[valid] = (
            np.exp(-1.0 / np.maximum(k2[valid] * Lw * Lw, 1e-12))
            / np.maximum(k2[valid] * k2[valid], 1e-18)
            * np.exp(-k2[valid] * l_small * l_small)
            * spread[valid]
        )
        P[0, 0] = 0.0
        rng = np.random.default_rng(int(seed))
        h0 = np.sqrt(np.maximum(P, 0.0) * 0.5) * (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        )
        h0[0, 0] = 0.0
        omega = np.sqrt(np.maximum(G * k, 0.0))
        omega[0, 0] = 0.0
        invk = np.divide(1.0, k, out=np.zeros_like(k), where=k > 1e-12)
        self._kx = kx.astype(np.float32)
        self._ky = ky.astype(np.float32)
        self._omega = omega.astype(np.float32)
        self._h0 = h0.astype(np.complex64)
        self._h0_conj_neg = np.conj(_negk(h0)).astype(np.complex64)
        self._ikx_over_k = (1j * kx * invk).astype(np.complex64)
        self._iky_over_k = (1j * ky * invk).astype(np.complex64)
        self.height = np.zeros((n, n), dtype=np.float32)
        self.dx = np.zeros((n, n), dtype=np.float32)
        self.dy = np.zeros((n, n), dtype=np.float32)
        self.sx = np.zeros((n, n), dtype=np.float32)
        self.sy = np.zeros((n, n), dtype=np.float32)
        self.vz = np.zeros((n, n), dtype=np.float32)
        self.evolve(0.0)
        rms = float(np.sqrt(np.mean(self.height * self.height)))
        if rms > 1e-12 and self.height_rms > 0.0:
            scale = np.float64(self.height_rms) / rms
            self._h0 *= scale
            self._h0_conj_neg *= scale
            self._t = None
            self.evolve(0.0)

    def evolve(self, t: float) -> None:
        t = float(t)
        if self._t is not None and abs(t - self._t) < 1e-6:
            return
        self._t = t
        phase = np.exp(-1j * self._omega * t)
        h = self._h0 * phase + self._h0_conj_neg * np.conj(phase)
        h[0, 0] = 0.0
        self.height = np.fft.ifft2(h).real.astype(np.float32)
        chop = float(self.choppiness)
        if chop > 1e-8:
            self.dx = (chop * np.fft.ifft2(-self._ikx_over_k * h).real).astype(np.float32)
            self.dy = (chop * np.fft.ifft2(-self._iky_over_k * h).real).astype(np.float32)
        else:
            self.dx.fill(0.0)
            self.dy.fill(0.0)
        self.sx = np.fft.ifft2(1j * self._kx * h).real.astype(np.float32)
        self.sy = np.fft.ifft2(1j * self._ky * h).real.astype(np.float32)
        dh = -1j * self._omega * (self._h0 * phase - self._h0_conj_neg * np.conj(phase))
        dh[0, 0] = 0.0
        self.vz = np.fft.ifft2(dh).real.astype(np.float32)

    def sample(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear wrap. Returns eta, dx, dy, sx, sy, vz at world (x, y)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        n = self.n
        s = n / self.tile_m
        u = xy[:, 0] * s
        v = xy[:, 1] * s
        u = np.mod(u, n)
        v = np.mod(v, n)
        i0 = np.floor(u).astype(np.int32)
        j0 = np.floor(v).astype(np.int32)
        su = (u - i0).astype(np.float32)
        sv = (v - j0).astype(np.float32)
        i0 = np.mod(i0, n)
        j0 = np.mod(j0, n)
        i1 = (i0 + 1) % n
        j1 = (j0 + 1) % n

        def _bilerp(img: np.ndarray) -> np.ndarray:
            a00 = img[j0, i0]
            a10 = img[j0, i1]
            a01 = img[j1, i0]
            a11 = img[j1, i1]
            return (a00 * (1.0 - su) + a10 * su) * (1.0 - sv) + (a01 * (1.0 - su) + a11 * su) * sv

        return (
            _bilerp(self.height),
            _bilerp(self.dx),
            _bilerp(self.dy),
            _bilerp(self.sx),
            _bilerp(self.sy),
            _bilerp(self.vz),
        )

    def normal_rgb_u8(self) -> np.ndarray:
        nx = -self.sx
        ny = -self.sy
        nz = np.ones_like(self.height)
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
        rgb = np.stack([nx * inv, ny * inv, nz * inv], axis=2)
        rgb = np.clip(rgb * 0.5 + 0.5, 0.0, 1.0)
        return (rgb * 255.0 + 0.5).astype(np.uint8)
