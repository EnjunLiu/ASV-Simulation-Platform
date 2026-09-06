"""Gerstner wave field (displacement + orbital kinematics) + optional Tessendorf FFT."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fft_ocean import TessendorfField


@dataclass
class WaveTrain:
    amplitude: float
    wavelength: float
    direction: tuple[float, float]
    steepness: float = 0.35
    phase: float = 0.0


# amp weight, λ m, direction, steepness, phase. Weights sum to 1; scaled to ~0.11 m.
_MILD_SEA = (
    (0.22, 15.0, (0.98, 0.10), 0.26, 0.31),
    (0.18, 11.5, (0.95, 0.32), 0.30, 1.84),
    (0.12, 10.2, (0.88, -0.18), 0.32, 4.02),
    (0.11, 8.6, (0.72, 0.69), 0.36, 2.17),
    (0.09, 7.4, (0.55, 0.84), 0.38, 5.41),
    (0.08, 6.3, (0.92, 0.40), 0.40, 0.77),
    (0.07, 5.4, (0.20, 0.98), 0.44, 3.33),
    (0.05, 4.7, (0.40, -0.92), 0.46, 1.55),
    (0.05, 4.1, (0.80, 0.60), 0.48, 4.90),
    (0.03, 3.7, (0.62, -0.78), 0.50, 2.62),
)
_MILD_SEA_AMP = 0.11


def mild_sea(amp_scale: float = 1.0) -> "WaveField":
    """Irregular long Gerstner (λ ≥ 3.6 m) plus Tessendorf FFT for 0.5–3.5 m."""
    s = float(amp_scale) * _MILD_SEA_AMP
    fft = None
    if float(amp_scale) > 1e-8:
        fft = TessendorfField(
            n=256,
            tile_m=64.0,
            lam_min=0.5,
            lam_max=3.5,
            height_rms=0.028 * float(amp_scale),
            choppiness=0.85,
            seed=7,
        )
    return WaveField(
        trains=[
            WaveTrain(frac * s, lam, direction, steep, phase)
            for frac, lam, direction, steep, phase in _MILD_SEA
        ],
        fft=fft,
    )


@dataclass
class WaveField:
    trains: list[WaveTrain] = field(default_factory=list)
    gravity: float = 9.81
    rest_z: float = 0.0
    fft: object = field(default=None, repr=False, compare=False)
    _gpu: object = field(default=None, repr=False, compare=False)

    def height(self, x: float, y: float, t: float) -> float:
        eta, _, _ = self.kinematics(np.array([[x, y]], dtype=np.float64), t)
        return float(eta[0])

    def heights(self, xy: np.ndarray, t: float) -> np.ndarray:
        eta, _, _ = self.kinematics(xy, t)
        return eta

    def kinematics(self, xy: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gerstner eta, orbital velocity, orbital acceleration at rest (x, y)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        n = int(xy.shape[0])
        eta = np.full(n, self.rest_z, dtype=np.float64)
        vel = np.zeros((n, 3), dtype=np.float64)
        acc = np.zeros((n, 3), dtype=np.float64)
        x, y = xy[:, 0], xy[:, 1]
        for w in self.trains:
            d = _dir(w)
            k, omega = _k_omega(w, self.gravity)
            a = float(w.amplitude)
            qi = float(np.clip(w.steepness, 0.0, 1.0)) * a
            theta = k * (d[0] * x + d[1] * y) - omega * t + w.phase
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            eta = eta + a * sin_t
            vel[:, 0] = vel[:, 0] + d[0] * qi * omega * sin_t
            vel[:, 1] = vel[:, 1] + d[1] * qi * omega * sin_t
            vel[:, 2] = vel[:, 2] - a * omega * cos_t
            acc[:, 0] = acc[:, 0] - d[0] * qi * omega * omega * cos_t
            acc[:, 1] = acc[:, 1] - d[1] * qi * omega * omega * cos_t
            acc[:, 2] = acc[:, 2] - a * omega * omega * sin_t
        fft = self.fft
        if fft is not None:
            fft.evolve(float(t))
            h, _dx, _dy, _sx, _sy, vz = fft.sample(xy)
            eta = eta + h.astype(np.float64)
            vel[:, 2] = vel[:, 2] + vz.astype(np.float64)
        return eta, vel, acc

    def deform(self, rest_xyz: np.ndarray, t: float) -> np.ndarray:
        pts, _nrm = self.deform_with_normals(rest_xyz, t)
        return pts

    def deform_with_normals(self, rest_xyz: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        rest = np.asarray(rest_xyz, dtype=np.float32).reshape(-1, 3)
        gpu = self._gpu
        if gpu is not None and int(getattr(gpu, "n", -1)) == int(rest.shape[0]):
            gpu.launch(t)
            return gpu.out.numpy(), gpu.nrm.numpy()
        return _deform_cpu(self, rest, t)

    def bind_grid(self, rest_xyz: np.ndarray) -> bool:
        """Upload rest vertices once. Later deforms are one Warp launch."""
        from .warp_gerstner import WarpGerstnerGrid, _warp

        if _warp() is None:
            self._gpu = None
            return False
        rest = np.asarray(rest_xyz, dtype=np.float64).reshape(-1, 3)
        self._gpu = WarpGerstnerGrid(self, rest)
        print("Warp Gerstner bound", rest.shape[0], "verts")
        return True


def _deform_cpu(field: WaveField, rest: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    out = rest.astype(np.float32, copy=True)
    out[:, 2] = field.rest_z
    x, y = rest[:, 0], rest[:, 1]
    nx = np.zeros(rest.shape[0], dtype=np.float32)
    ny = np.zeros(rest.shape[0], dtype=np.float32)
    nz = np.ones(rest.shape[0], dtype=np.float32)
    for w in field.trains:
        d = _dir(w)
        k, omega = _k_omega(w, field.gravity)
        a = float(w.amplitude)
        qi = float(np.clip(w.steepness, 0.0, 1.0)) * a
        theta = k * (d[0] * x + d[1] * y) - omega * t + w.phase
        cos_t = np.cos(theta).astype(np.float32)
        sin_t = np.sin(theta).astype(np.float32)
        out[:, 0] = out[:, 0] + np.float32(d[0] * qi) * cos_t
        out[:, 1] = out[:, 1] + np.float32(d[1] * qi) * cos_t
        out[:, 2] = out[:, 2] + np.float32(a) * sin_t
        nx = nx - np.float32(d[0] * a * k) * cos_t
        ny = ny - np.float32(d[1] * a * k) * cos_t
        nz = nz - np.float32(qi * k) * sin_t
    fft = field.fft
    if fft is not None:
        fft.evolve(float(t))
        h, dx, dy, sx, sy, _vz = fft.sample(np.stack([x, y], axis=1))
        out[:, 0] = out[:, 0] + dx.astype(np.float32)
        out[:, 1] = out[:, 1] + dy.astype(np.float32)
        out[:, 2] = out[:, 2] + h.astype(np.float32)
        nx = nx - sx.astype(np.float32)
        ny = ny - sy.astype(np.float32)
    nrm = np.stack([nx, ny, nz], axis=1)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return out, nrm


def _dir(w: WaveTrain) -> np.ndarray:
    d = np.array([float(w.direction[0]), float(w.direction[1])], dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return d / n


def _k_omega(w: WaveTrain, gravity: float) -> tuple[float, float]:
    k = 2.0 * np.pi / max(float(w.wavelength), 1e-6)
    omega = float(np.sqrt(max(gravity * k, 0.0)))
    return k, omega



