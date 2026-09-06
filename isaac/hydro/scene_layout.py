"""HIL SceneAuto layouts (UE centimetres) mapped into Isaac metres.

Source geometry is mirrored by the sibling Unreal Engine implementation.
MakeLayout(). L7B and T1 use the same fixed multi-vessel scene coordinates.

UE is left-handed, X forward, Y right, centimetres.
Isaac / this repo is Z-up, body X forward, Y left, metres.
"""

from __future__ import annotations

from dataclasses import dataclass

# HIL target boats. Two whites match BP_Target2 / BP_Target3; gray is not in this scene.
HIL_TARGETS = (
    ("target_red", "red", "TargetRed"),
    ("target_blue", "blue", "TargetBlue"),
    ("target_left", "white", "TargetLeft"),
    ("target_right", "white", "TargetRight"),
)

# UE centimetres, X forward, Y right. Ego ASV is at the origin, yaw 0.
HIL_LAYOUTS_UE_CM = {
    "L1": {
        "target_red": (150.0, 0.0),
        "target_blue": (400.0, 0.0),
        "target_left": (250.0, -150.0),
        "target_right": (250.0, 150.0),
    },
    "L2": {
        "target_red": (400.0, 0.0),
        "target_blue": (150.0, 0.0),
        "target_left": (250.0, -150.0),
        "target_right": (250.0, 150.0),
    },
    "L3": {
        "target_red": (250.0, -120.0),
        "target_blue": (250.0, 120.0),
        "target_left": (150.0, -180.0),
        "target_right": (400.0, 180.0),
    },
    "L4": {
        "target_red": (250.0, 120.0),
        "target_blue": (250.0, -120.0),
        "target_left": (400.0, -180.0),
        "target_right": (150.0, 180.0),
    },
    "L5": {
        "target_red": (600.0, 0.0),
        "target_blue": (900.0, 0.0),
        "target_left": (900.0, -300.0),
        "target_right": (900.0, 300.0),
    },
    "L6": {
        "target_red": (2500.0, -300.0),
        "target_blue": (2500.0, 300.0),
        "target_left": (3500.0, -800.0),
        "target_right": (3500.0, 800.0),
    },
    "L6B": {
        "target_red": (2500.0, 300.0),
        "target_blue": (2500.0, -300.0),
        "target_left": (3500.0, -800.0),
        "target_right": (3500.0, 800.0),
    },
    "L7": {
        "target_red": (450.0, -100.0),
        "target_blue": (450.0, 100.0),
        "target_left": (700.0, -350.0),
        "target_right": (700.0, 350.0),
    },
    "L7B": {
        "target_red": (450.0, 100.0),
        "target_blue": (450.0, -100.0),
        "target_left": (700.0, -350.0),
        "target_right": (700.0, 350.0),
    },
}

DEFAULT_LAYOUT = "L7B"
DEFAULT_FOLLOW_ENTITY = "target_red"
LAYOUT_IDS = tuple(HIL_LAYOUTS_UE_CM.keys())


@dataclass(frozen=True)
class TargetPose:
    entity_id: str
    color: str
    prim_name: str
    x: float
    y: float
    ue_x_cm: float
    ue_y_cm: float


def isaac_xy_from_ue_cm(x_cm: float, y_cm: float) -> tuple[float, float]:
    """UE cm (X forward, Y right) -> Isaac m (X forward, Y left)."""
    return float(x_cm) / 100.0, -float(y_cm) / 100.0


def layout_poses(layout_id: str = DEFAULT_LAYOUT) -> list[TargetPose]:
    key = str(layout_id).strip().upper()
    table = HIL_LAYOUTS_UE_CM.get(key)
    if table is None:
        raise ValueError(f"unknown HIL layout {layout_id!r}; expected {LAYOUT_IDS}")
    poses = []
    for entity_id, color, prim_name in HIL_TARGETS:
        ue_x, ue_y = table[entity_id]
        x, y = isaac_xy_from_ue_cm(ue_x, ue_y)
        poses.append(
            TargetPose(
                entity_id=entity_id,
                color=color,
                prim_name=prim_name,
                x=x,
                y=y,
                ue_x_cm=ue_x,
                ue_y_cm=ue_y,
            )
        )
    return poses


def target_prim_path(prim_name: str) -> str:
    return f"/World/{prim_name}"


def target_root_path(prim_name: str) -> str:
    return f"/World/{prim_name}/ASV_Root"


# howto 1.1: /asv/targets/{red,blue,white_l,white_r}/odom
TARGET_ODOM_SLUG = {
    "target_red": "red",
    "target_blue": "blue",
    "target_left": "white_l",
    "target_right": "white_r",
}


def target_odom_topic(entity_id: str, namespace: str = "asv") -> str:
    ns = str(namespace).strip("/")
    slug = TARGET_ODOM_SLUG[entity_id]
    return f"/{ns}/targets/{slug}/odom"
