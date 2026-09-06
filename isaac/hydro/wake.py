"""Visual Kelvin wake. Not foam, not physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_KELVIN_TAN = 0.353553  # tan(19.47°)


@dataclass
class ShipWake:
    x: float = 0.0
    y: float = 0.0
    hx: float = 1.0
    hy: float = 0.0
    speed: float = 0.0
    length: float = 1.4
    beam: float = 0.4
    amp: float = 0.045
    gravity: float = 9.81

    def normalized(self) -> "ShipWake":
        h = np.array([float(self.hx), float(self.hy)], dtype=np.float64)
        n = float(np.linalg.norm(h))
        if n < 1e-8:
            h = np.array([1.0, 0.0])
        else:
            h = h / n
        return ShipWake(
            x=float(self.x),
            y=float(self.y),
            hx=float(h[0]),
            hy=float(h[1]),
            speed=max(float(self.speed), 0.0),
            length=max(float(self.length), 0.2),
            beam=max(float(self.beam), 0.1),
            amp=float(self.amp),
            gravity=float(self.gravity),
        )


def wake_height(xy: np.ndarray, wake: ShipWake) -> np.ndarray:
    """eta at world XY. Zero in front; V-wake + transverse behind; small bow mound."""
    w = wake.normalized()
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    rx = (xy[:, 0] - w.x) * w.hx + (xy[:, 1] - w.y) * w.hy
    ry = -(xy[:, 0] - w.x) * w.hy + (xy[:, 1] - w.y) * w.hx
    u = float(w.speed)
    if u < 0.05:
        return np.zeros(xy.shape[0], dtype=np.float64)
    L = float(w.length)
    B = float(w.beam)
    aft = -rx
    k = w.gravity / max(u * u, 0.15)
    lam = 2.0 * np.pi / max(k, 1e-3)
    amp = float(w.amp) * min((u / 0.8) ** 2, 2.5)
    behind = aft >= -0.15 * L
    sigma = 0.18 * L + 0.10 * np.maximum(aft, 0.0)
    arm = np.abs(ry) - _KELVIN_TAN * np.maximum(aft, 0.0)
    env = np.exp(-np.maximum(aft, 0.0) / (9.0 * L)) * np.exp(-(arm * arm) / (sigma * sigma + 1e-6))
    env = env * np.exp(-(ry * ry) / ((2.4 * B + 0.25 * np.maximum(aft, 0.0)) ** 2 + 1e-6))
    trans = np.sin(2.0 * np.pi * aft / lam)
    diverg = np.sin(2.0 * np.pi * np.hypot(aft, ry) / (0.7 * lam))
    bow = np.exp(-(rx - 0.35 * L) ** 2 / (0.18 * L) ** 2) * np.exp(-(ry * ry) / (0.35 * B) ** 2)
    eta = amp * env * (0.65 * trans + 0.35 * diverg)
    eta = eta + 0.35 * amp * bow
    eta = np.where(behind | (bow > 0.02), eta, 0.0)
    far = np.hypot(aft, ry) > 18.0 * L
    eta = np.where(far, 0.0, eta)
    return eta.astype(np.float64)
