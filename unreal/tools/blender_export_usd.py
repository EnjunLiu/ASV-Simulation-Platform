"""Export one Blender file to a versioned USD asset and write its manifest.

Meshes are joined into a single object before export so UE can overwrite one
StaticMesh. The source .blend is not saved.

Run with Blender's bundled Python, for example:
  blender.exe --background --python tools/blender_export_usd.py -- \
    --source-blend C:\\assets\\SM_ASV.blend \
    --asset-root C:\\Users\\LIU\\Desktop\\ASVModels \
    --asset-name SM_ASV \
    --ue-destination /Game/Assets/ASV/_staging
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from asset_pipeline_common import (  # noqa: E402
    DEFAULT_MAX_KEPT_VERSIONS,
    DEFAULT_STAGING,
    next_version,
    prune_old_versions,
    sha256_file,
    stem_to_asset_name,
    ue_target_mesh_for_name,
    validate_version,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--asset-name", help="Defaults to the blend file stem")
    parser.add_argument("--ue-destination", default=DEFAULT_STAGING)
    parser.add_argument("--ue-target-mesh", help="UE StaticMesh path to overwrite")
    parser.add_argument("--version", help="Explicit version such as v002")
    parser.add_argument("--max-kept-versions", type=int, default=DEFAULT_MAX_KEPT_VERSIONS)
    return parser.parse_args(argv)


def _join_visible_meshes() -> int:
    view_layer = bpy.context.view_layer
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_viewport and not obj.hide_get()
    ]
    if not meshes:
        raise RuntimeError("Blender file contains no visible mesh objects")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(True)
    view_layer.objects.active = meshes[0]
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    if len(meshes) > 1:
        bpy.ops.object.join()
    return len(meshes)


def _export_usd(path: Path, root_prim: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    options = {
        "filepath": str(path),
        "selected_objects_only": True,
        "visible_objects_only": True,
        "export_animation": False,
        "export_uvmaps": True,
        "export_normals": True,
        "export_materials": True,
        "export_textures": True,
        "use_instancing": False,
        "evaluation_mode": "DAG_EVAL_VIEWPORT",
        "generate_preview_surface": True,
        "export_custom_properties": True,
        "relative_paths": True,
        "root_prim_path": root_prim,
        "meters_per_unit": 1.0,
    }
    try:
        bpy.ops.wm.usd_export(**options)
    except TypeError:
        fallback = {
            key: options[key]
            for key in (
                "filepath",
                "selected_objects_only",
                "export_animation",
                "export_uvmaps",
                "export_normals",
                "export_materials",
            )
        }
        bpy.ops.wm.usd_export(**fallback)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Blender USD export did not create a file: {path}")


def main() -> int:
    args = _parse_args()
    source_blend = args.source_blend.resolve()
    asset_root = args.asset_root.resolve()
    if not source_blend.is_file():
        raise FileNotFoundError(f"source blend not found: {source_blend}")

    asset_name = stem_to_asset_name(args.asset_name or source_blend.stem)
    ue_target_mesh = args.ue_target_mesh or ue_target_mesh_for_name(asset_name)
    published_dir = asset_root / "published"
    published_dir.mkdir(parents=True, exist_ok=True)
    version = validate_version(args.version) if args.version else next_version(published_dir)
    version_dir = published_dir / version
    if version_dir.exists() and any(version_dir.iterdir()):
        raise FileExistsError(f"version already contains files: {version_dir}")
    version_dir.mkdir(parents=True, exist_ok=True)

    usd_path = version_dir / f"{asset_name}_{version}.usdc"
    bpy.ops.wm.open_mainfile(filepath=str(source_blend), load_ui=False)
    source_mesh_count = _join_visible_meshes()
    _export_usd(usd_path, f"/{asset_name}")

    manifest = {
        "schema_version": 2,
        "asset_name": asset_name,
        "version": version,
        "status": "published",
        "source_blend": str(source_blend),
        "usd_file": usd_path.name,
        "usd_path": str(usd_path),
        "ue_destination": args.ue_destination.rstrip("/"),
        "ue_target_mesh": ue_target_mesh,
        "unit": "meter",
        "materials_verified": False,
        "max_kept_versions": args.max_kept_versions,
        "source_mesh_count": source_mesh_count,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {usd_path.name: sha256_file(usd_path)},
        "exporter": {
            "name": "Blender USD exporter",
            "blender_version": bpy.app.version_string,
        },
    }
    manifest_path = version_dir / "asset_manifest.json"
    write_json(manifest_path, manifest)
    removed = prune_old_versions(published_dir, args.max_kept_versions)
    print(
        __import__("json").dumps(
            {
                "usd": str(usd_path),
                "manifest": str(manifest_path),
                "ue_target_mesh": ue_target_mesh,
                "pruned": removed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
