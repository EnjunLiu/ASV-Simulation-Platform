"""Shared contract for the Blender -> Kit -> UE5 asset pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

SCHEMA_VERSION = 2
DEFAULT_MAX_KEPT_VERSIONS = 5
DEFAULT_CONTENT_FOLDER = "/Game/ASVModel"
DEFAULT_STAGING = "/Game/Assets/ASV/_staging"

# One-time aliases for files that predate the English naming contract.
ASSET_NAME_ALIASES = {
    "ASVModel": "SM_ASV",
    "ASV静态网格体": "SM_ASV",
}


def stem_to_asset_name(stem: str) -> str:
    return ASSET_NAME_ALIASES.get(stem, stem)


def ue_target_mesh_for_name(asset_name: str) -> str:
    name = stem_to_asset_name(asset_name)
    return f"{DEFAULT_CONTENT_FOLDER.rstrip('/')}/{name}"


def validate_version(value: str) -> str:
    if not re.fullmatch(r"v[0-9]{3}", value):
        raise ValueError("version must match v001, v002, ...")
    return value


def next_version(published_dir: Path) -> str:
    versions = []
    for child in published_dir.glob("v[0-9][0-9][0-9]"):
        try:
            versions.append(int(child.name[1:]))
        except ValueError:
            continue
    return f"v{(max(versions, default=0) + 1):03d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_versions(published_dir: Path) -> list[str]:
    versions = []
    for child in published_dir.glob("v[0-9][0-9][0-9]"):
        if child.is_dir():
            try:
                versions.append((int(child.name[1:]), child.name))
            except ValueError:
                continue
    versions.sort()
    return [name for _, name in versions]


def preview_prune(published_dir: Path, keep: int, extra_new: int = 1) -> list[str]:
    names = list_versions(published_dir)
    after_count = len(names) + extra_new
    drop = max(0, after_count - keep)
    return names[:drop]


def prune_old_versions(published_dir: Path, keep: int) -> list[str]:
    if keep < 1:
        raise ValueError("max_kept_versions must be >= 1")
    versions = []
    for child in published_dir.glob("v[0-9][0-9][0-9]"):
        if not child.is_dir():
            continue
        try:
            versions.append((int(child.name[1:]), child))
        except ValueError:
            continue
    versions.sort()
    removed = []
    for _, folder in versions[:-keep]:
        shutil.rmtree(folder)
        removed.append(folder.name)
    return removed
