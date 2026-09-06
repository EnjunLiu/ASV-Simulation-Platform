"""Water column. sample_height is still water or a Warp/Gerstner field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WaterState:
    rho: float = 1000.0
    gravity: float = 9.81
    water_z: float = 0.0
    waves: Optional[object] = None
    time: float = 0.0

    def sample_height(self, x: float, y: float) -> float:
        eta, _, _ = self.sample_kinematics([[x, y]])
        return float(eta[0])

    def sample_heights(self, xy) -> np.ndarray:
        eta, _, _ = self.sample_kinematics(xy)
        return eta

    def sample_kinematics(self, xy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        n = int(xy.shape[0])
        if self.waves is None:
            eta = np.full(n, float(self.water_z), dtype=np.float64)
            z = np.zeros((n, 3), dtype=np.float64)
            return eta, z, z.copy()
        return self.waves.kinematics(xy, self.time)
