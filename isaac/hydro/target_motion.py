"""Target planar motion (howto 1.1.3). Heave/roll/pitch stay with hydro."""

from __future__ import annotations

import math

MOTIONS = ("S0", "S1", "S2")
S2_SURGE = 0.6
S2_WAVELENGTH = 60.0
S2_AMP_RB = 3.0
S2_AMP_WHITE = 0.0

# World XY m/s before Dir. Red/blue scale Y by Dir; whites scale X by Dir.
S1_VEL = {
    "target_red": (0.0, -0.12),
    "target_blue": (0.0, 0.09),
    "target_left": (0.07, -0.03),
    "target_right": (-0.05, 0.04),
}

S2_AMP = {
    "target_red": S2_AMP_RB,
    "target_blue": S2_AMP_RB,
    "target_left": S2_AMP_WHITE,
    "target_right": S2_AMP_WHITE,
}


def parse_motion(text: str) -> str:
    key = str(text or "S0").strip().upper()
    if key not in MOTIONS:
        raise ValueError(f"unknown --motion {text!r}; expected {MOTIONS}")
    return key


def motion_dir(seed: int) -> int:
    """Even seed → +1, odd → −1."""
    return 1 if int(seed) % 2 == 0 else -1


def motion_phase(seed: int) -> float:
    """S2 sinusoid phase (rad). One phase from the seed, shared by all targets."""
    return (int(seed) % 360) * math.pi / 180.0


def s1_velocity(entity_id: str, seed: int) -> tuple[float, float]:
    vx, vy = S1_VEL[entity_id]
    d = float(motion_dir(seed))
    if entity_id in ("target_red", "target_blue"):
        return vx, vy * d
    return vx * d, vy


def s2_velocity(entity_id: str, seed: int, t: float, amp: float | None = None) -> tuple[float, float]:
    a = S2_AMP[entity_id] if amp is None else float(amp)
    tt = max(0.0, float(t))
    omega = 2.0 * math.pi * S2_SURGE / S2_WAVELENGTH
    phi = motion_phase(seed)
    vx = S2_SURGE
    vy = -a * omega * math.cos(omega * tt + phi)
    return vx, vy


def s2_position(
    entity_id: str,
    seed: int,
    t: float,
    x0: float,
    y0: float,
    amp: float | None = None,
) -> tuple[float, float]:
    a = S2_AMP[entity_id] if amp is None else float(amp)
    tt = max(0.0, float(t))
    omega = 2.0 * math.pi * S2_SURGE / S2_WAVELENGTH
    phi = motion_phase(seed)
    x = float(x0) + S2_SURGE * tt
    y = float(y0) - a * math.sin(omega * tt + phi)
    return x, y


def planar_cmd(motion: str, entity_id: str, seed: int, t: float) -> tuple[float, float, float]:
    """Return (vx, vy, yaw). Yaw from planar velocity; still → 0."""
    key = parse_motion(motion)
    if key == "S0":
        return 0.0, 0.0, 0.0
    if key == "S1":
        vx, vy = s1_velocity(entity_id, seed)
    else:
        vx, vy = s2_velocity(entity_id, seed, t)
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        return 0.0, 0.0, 0.0
    return vx, vy, math.atan2(vy, vx)


def planar_cmd_armed(
    motion: str,
    entity_id: str,
    seed: int,
    t: float,
    spin_delay: float = 0.0,
) -> tuple[float, float, float]:
    """Hold still until spin_delay. S2 at t=0 is already 0.6 m/s; do not apply that during settle."""
    delay = float(spin_delay)
    if float(t) < delay:
        return 0.0, 0.0, 0.0
    return planar_cmd(motion, entity_id, seed, float(t) - delay)
