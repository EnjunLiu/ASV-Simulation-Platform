"""ASV USD helpers: color variants and Blender leftover mesh names."""

from __future__ import annotations

import os
import re

_MESH_INDEX = re.compile(r"^Mesh_\d+$")

ASV_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "asv_models")
DEFAULT_USD = os.path.join(ASV_MODELS_DIR, "ASVModel.usd")
VARIANTS_DIR = os.path.join(ASV_MODELS_DIR, "variants")

# Yellow is the authored source file. Do not emit a yellow payload.
HULL_COLORS = ("yellow", "red", "blue", "white", "gray")


def usd_for_hull_color(color: str, *, models_dir: str | None = None) -> str:
    name = str(color).strip().lower()
    root = models_dir or ASV_MODELS_DIR
    if name in ("", "yellow"):
        return os.path.join(root, "ASVModel.usd")
    if name not in HULL_COLORS:
        raise ValueError(f"unknown hull color {color!r}; expected {HULL_COLORS}")
    return os.path.join(root, "variants", f"ASVModel_{name}.usda")


def suggested_mesh_name(parent_name: str, mesh_name: str) -> str | None:
    """Human name for a leftover Blender mesh. None = keep.

    Names are authored in ASVModel.blend / ASVModel.usd. Do not rename
    referenced prims at Isaac load (MovePrim fails on ancestral refs).
    """
    parent = str(parent_name)
    mesh = str(mesh_name)
    if not parent or not mesh:
        return None
    if _MESH_INDEX.match(mesh):
        return f"{parent}_Mesh"
    if parent.endswith("_Plate") and mesh.endswith("_Mesh") and not mesh.startswith(parent):
        return f"{parent}_Mesh"
    return None


def find_hull_mesh_path(stage, root_path: str) -> str:
    """Hull triangle mesh under `{root}/Hull`. Prefers Hull_Mesh over Mesh_006."""
    hull_xform = f"{root_path.rstrip('/')}/Hull"
    preferred = f"{hull_xform}/Hull_Mesh"
    prim = stage.GetPrimAtPath(preferred)
    if prim.IsValid():
        return preferred
    leftover = f"{hull_xform}/Mesh_006"
    prim = stage.GetPrimAtPath(leftover)
    if prim.IsValid():
        return leftover
    hull = stage.GetPrimAtPath(hull_xform)
    if hull.IsValid():
        from pxr import UsdGeom

        for child in hull.GetChildren():
            if child.IsA(UsdGeom.Mesh):
                return str(child.GetPath())
    raise RuntimeError(f"hull mesh not found under {hull_xform}")
