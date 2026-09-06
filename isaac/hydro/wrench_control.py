"""Pure-math helpers for the force/moment ASV HIL path."""

from __future__ import annotations

from dataclasses import dataclass
import math

EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class MixedThrust:
    left: float
    right: float
    force: float
    moment: float
    saturated: bool


def mix_force_moment(
    force: float,
    moment: float,
    *,
    half_spacing: float = 0.11,
    max_thrust: float = 3.0,
) -> MixedThrust:
    """Map body surge force and +Z yaw moment to two reversible propellers."""
    d = abs(float(half_spacing))
    limit = abs(float(max_thrust))
    if d < 1.0e-6 or limit <= 0.0:
        return MixedThrust(0.0, 0.0, 0.0, 0.0, True)
    f = float(force)
    n = float(moment)
    if not math.isfinite(f) or not math.isfinite(n):
        return MixedThrust(0.0, 0.0, 0.0, 0.0, True)
    raw_left = 0.5 * (f - n / d)
    raw_right = 0.5 * (f + n / d)
    left = max(-limit, min(limit, raw_left))
    right = max(-limit, min(limit, raw_right))
    return MixedThrust(
        left=left,
        right=right,
        force=left + right,
        moment=d * (right - left),
        saturated=abs(left - raw_left) > 1.0e-12 or abs(right - raw_right) > 1.0e-12,
    )


def world_xy_to_navsat(
    x_east: float,
    y_north: float,
    z_up: float,
    *,
    origin_latitude: float = 31.2304,
    origin_longitude: float = 121.4737,
    origin_altitude: float = 0.0,
) -> tuple[float, float, float]:
    """Small-area ENU to WGS84 conversion used by the simulated GNSS."""
    lat0 = float(origin_latitude)
    latitude = lat0 + math.degrees(float(y_north) / EARTH_RADIUS_M)
    longitude = float(origin_longitude) + math.degrees(
        float(x_east) / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))
    )
    return latitude, longitude, float(origin_altitude) + float(z_up)


class WrenchWatchdog:
    def __init__(self, timeout_s: float = 0.25):
        self.timeout_s = float(timeout_s)
        self.force = 0.0
        self.moment = 0.0
        self.stamp = -1.0

    def update(self, force: float, moment: float, stamp: float) -> bool:
        values = (float(force), float(moment), float(stamp))
        if not all(math.isfinite(v) for v in values) or values[2] <= self.stamp:
            return False
        self.force, self.moment, self.stamp = values
        return True

    def command(self, now: float) -> tuple[float, float, bool]:
        fresh = self.stamp >= 0.0 and 0.0 <= float(now) - self.stamp <= self.timeout_s
        return (self.force, self.moment, True) if fresh else (0.0, 0.0, False)
