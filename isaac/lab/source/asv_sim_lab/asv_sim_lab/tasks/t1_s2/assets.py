"""Portable paths for local, Git-ignored policy assets."""

from __future__ import annotations

import os
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[5]
RUNTIME_ASSETS = LAB_ROOT / "runtime_assets"


def asset_path(environment_name: str, filename: str) -> str:
    return os.environ.get(environment_name, str(RUNTIME_ASSETS / filename))


APPEARANCE_LIBRARY = asset_path("ASV_LAB_APPEARANCE_LIBRARY", "owl512_appearance_library_v1.npz")
TASK_EMBEDDINGS = asset_path("ASV_LAB_TASK_EMBEDDINGS", "qwen_task_embed.npz")
ACTOR_CHECKPOINT = asset_path(
    "ASV_LAB_ACTOR_CHECKPOINT", "actor_bc_semantic16_v13_dagger_polar_frozen.pt"
)
