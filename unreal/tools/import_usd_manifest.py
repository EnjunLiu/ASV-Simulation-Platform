"""Import a published USD asset and overwrite the StaticMesh named in the manifest.

Does not spawn level actors and does not save Main_Map.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import unreal

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from asset_pipeline_common import (  # noqa: E402
    load_manifest,
    sha256_file,
    write_json,
)


def _manifest_path() -> Path:
    value = os.environ.get("ASSET_PIPELINE_MANIFEST") or os.environ.get("ASV_ASSET_MANIFEST")
    if not value:
        raise RuntimeError("Set ASSET_PIPELINE_MANIFEST to an asset_manifest.json path")
    return Path(value).expanduser().resolve()


def _imported_static_meshes(imported_paths: list[str]) -> list[unreal.StaticMesh]:
    meshes = []
    seen = set()
    for object_path in imported_paths:
        asset = unreal.load_asset(object_path)
        if not isinstance(asset, unreal.StaticMesh):
            continue
        path_name = asset.get_path_name()
        if path_name in seen:
            continue
        seen.add(path_name)
        meshes.append(asset)
    return meshes


def _largest_mesh(meshes: list[unreal.StaticMesh]) -> unreal.StaticMesh:
    def triangle_count(mesh: unreal.StaticMesh) -> int:
        try:
            return int(mesh.get_num_triangles(0))
        except Exception:
            try:
                return int(mesh.get_editor_property("num_triangles"))
            except Exception:
                return 0

    return max(meshes, key=triangle_count)


def _package_path(asset: unreal.Object) -> str:
    path_name = asset.get_path_name()
    return path_name.split(".", 1)[0]


def _restore_collision_flag(mesh: unreal.StaticMesh, flag) -> None:
    if flag is None:
        return
    body_setup = mesh.get_editor_property("body_setup")
    if not body_setup:
        return
    body_setup.set_editor_property("collision_trace_flag", flag)


# Slot names stayed Chinese after the English rename. Redirectors were deleted
# with the old folder, so the mesh must be rebound to the English materials.
SM_ASV_SLOT_MATERIALS = {
    "把手黑": "/Game/ASVModel/M_HandleBlack",
    "屏幕绿": "/Game/ASVModel/M_DisplayGreen",
    "底板黄": "/Game/ASVModel/M_BasePlateYellow",
    "按钮灰": "/Game/ASVModel/M_ButtonGray",
    "船身黄": "/Game/ASVModel/M_HullYellow",
    "船身白": "/Game/ASVModel/M_HullWhite",
    "螺丝银": "/Game/ASVModel/M_ScrewSilver",
    "螺旋桨蓝": "/Game/ASVModel/M_PropellerBlue",
}

SM_TARGET_SLOT_MATERIALS = {
    "把手黑": "/Game/ASVModel/M_HandleBlack",
    "屏幕绿": "/Game/ASVModel/M_DisplayGreen",
    "按钮灰": "/Game/ASVModel/M_ButtonGray",
    "底板黄": "/Game/ASVModel/M_BasePlateRed",
    "船身黄": "/Game/ASVModel/M_HullRed",
    "船身白": "/Game/ASVModel/M_HullWhite",
    "螺丝银": "/Game/ASVModel/M_ScrewSilver",
    "螺旋桨蓝": "/Game/ASVModel/M_PropellerBlue",
}

KNOWN_SLOT_MATERIALS = {
    "/Game/ASVModel/SM_ASV": SM_ASV_SLOT_MATERIALS,
    "/Game/ASVModel/SM_Target": SM_TARGET_SLOT_MATERIALS,
}


def restore_slot_materials(mesh_path: str, mapping: dict[str, str] | None = None) -> dict:
    """Rebind named material slots. Does not save Main_Map."""
    mapping = mapping or KNOWN_SLOT_MATERIALS.get(mesh_path, {})
    mesh = unreal.load_asset(mesh_path)
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"not a StaticMesh: {mesh_path}")
    assigned: dict[str, str] = {}
    skipped: list[str] = []
    static_mats = list(mesh.get_editor_property("static_materials") or [])
    for index, slot in enumerate(static_mats):
        slot_name = str(slot.get_editor_property("material_slot_name"))
        mat_path = mapping.get(slot_name)
        if not mat_path:
            skipped.append(slot_name)
            continue
        material = unreal.load_asset(mat_path)
        if material is None:
            skipped.append(slot_name)
            continue
        mesh.set_material(index, material)
        assigned[slot_name] = mat_path
    if assigned:
        unreal.EditorAssetLibrary.save_asset(mesh_path)
    return {"mesh": mesh_path, "assigned": assigned, "skipped": skipped}


MESH_BLUEPRINTS = {
    "/Game/ASVModel/SM_ASV": ("/Game/BP_ASV",),
    "/Game/ASVModel/SM_Target": (
        "/Game/BP_Target",
        "/Game/BP_Target1",
        "/Game/BP_Target2",
        "/Game/BP_Target3",
    ),
}


def _assign_mesh_if_needed(comp: unreal.StaticMeshComponent, mesh_path: str, mesh: unreal.StaticMesh) -> bool:
    current = comp.get_editor_property("static_mesh")
    current_path = ""
    if current is not None:
        current_path = current.get_path_name().split(".", 1)[0]
    if current is mesh:
        return False
    if current is None or current_path == mesh_path:
        comp.set_editor_property("static_mesh", mesh)
        return True
    return False


def _bind_blueprints_to_mesh(mesh_path: str, mesh: unreal.StaticMesh) -> int:
    """Assign mesh onto Blueprint CDOs/SCS templates. Does not save Main_Map."""
    replaced = 0
    for bp_path in MESH_BLUEPRINTS.get(mesh_path, ()):
        if not unreal.EditorAssetLibrary.does_asset_exist(bp_path):
            continue
        changed = False
        name = bp_path.rsplit("/", 1)[-1]
        cls = unreal.load_object(None, f"{bp_path}.{name}_C")
        if cls is not None:
            cdo = unreal.get_default_object(cls)
            for comp in cdo.get_components_by_class(unreal.StaticMeshComponent):
                if _assign_mesh_if_needed(comp, mesh_path, mesh):
                    changed = True
                    replaced += 1
        bp = unreal.load_asset(bp_path)
        try:
            scs = bp.get_editor_property("simple_construction_script")
            if scs:
                for node in scs.get_all_nodes():
                    template = node.get_editor_property("component_template")
                    if isinstance(template, unreal.StaticMeshComponent) and _assign_mesh_if_needed(
                        template, mesh_path, mesh
                    ):
                        changed = True
                        replaced += 1
        except Exception:
            pass
        if changed and bp is not None:
            try:
                unreal.BlueprintEditorLibrary.compile_blueprint(bp)
            except Exception:
                pass
            unreal.EditorAssetLibrary.save_asset(bp_path)
    return replaced


def _overwrite_target(source_mesh: unreal.StaticMesh, target_path: str) -> str:
    editor_assets = unreal.EditorAssetLibrary
    source_path = _package_path(source_mesh)
    if source_path == target_path:
        _bind_blueprints_to_mesh(target_path, source_mesh)
        return target_path

    dest_dir = "/".join(target_path.split("/")[:-1])
    if not editor_assets.does_directory_exist(dest_dir):
        editor_assets.make_directory(dest_dir)
    editor_assets.save_asset(source_path)

    if not editor_assets.does_asset_exist(target_path):
        if not editor_assets.rename_asset(source_path, target_path):
            raise RuntimeError(f"Failed to rename {source_path} -> {target_path}")
        live = unreal.load_asset(target_path)
        if isinstance(live, unreal.StaticMesh):
            _bind_blueprints_to_mesh(target_path, live)
        return target_path

    target_asset = unreal.load_asset(target_path)
    if isinstance(target_asset, unreal.StaticMesh):
        if editor_assets.consolidate_assets(source_mesh, [target_asset]):
            if source_path != target_path and editor_assets.does_asset_exist(source_path):
                editor_assets.rename_asset(source_path, target_path)
            return target_path

    temp_path = f"{target_path}_IMPORTED"
    if editor_assets.does_asset_exist(temp_path):
        editor_assets.delete_asset(temp_path)
    if not editor_assets.duplicate_asset(source_path, temp_path):
        raise RuntimeError(f"Failed to duplicate {source_path} -> {temp_path}")
    imported = unreal.load_asset(temp_path)
    if not isinstance(imported, unreal.StaticMesh):
        raise RuntimeError(f"duplicate did not produce a StaticMesh at {temp_path}")

    _replace_blueprint_meshes(target_path, imported)
    editor_assets.save_asset(temp_path)

    backup = f"{target_path}_OLD"
    if editor_assets.does_asset_exist(backup):
        editor_assets.delete_asset(backup)
    if editor_assets.rename_asset(target_path, backup) and editor_assets.rename_asset(temp_path, target_path):
        live = unreal.load_asset(target_path)
        if isinstance(live, unreal.StaticMesh):
            _replace_blueprint_meshes(backup, live)
        if editor_assets.does_asset_exist(backup):
            editor_assets.delete_asset(backup)
        return target_path
    return temp_path


def _replace_blueprint_meshes(old_path: str, new_mesh: unreal.StaticMesh) -> int:
    """Point Blueprint CDOs at new_mesh. Does not edit or save Main_Map."""
    replaced = 0
    old_mesh = None
    if unreal.EditorAssetLibrary.does_asset_exist(old_path):
        loaded = unreal.load_asset(old_path)
        if isinstance(loaded, unreal.StaticMesh):
            old_mesh = loaded
    for bp_path in (
        "/Game/BP_ASV",
        "/Game/BP_Target",
        "/Game/BP_Target1",
        "/Game/BP_Target2",
        "/Game/BP_Target3",
    ):
        if not unreal.EditorAssetLibrary.does_asset_exist(bp_path):
            continue
        name = bp_path.rsplit("/", 1)[-1]
        cls = unreal.load_object(None, f"{bp_path}.{name}_C")
        if cls is None:
            continue
        cdo = unreal.get_default_object(cls)
        changed = False
        for comp in cdo.get_components_by_class(unreal.StaticMeshComponent):
            current = comp.get_editor_property("static_mesh")
            current_path = ""
            if current is not None:
                current_path = current.get_path_name().split(".", 1)[0]
            if current is old_mesh or current_path == old_path:
                comp.set_editor_property("static_mesh", new_mesh)
                changed = True
                replaced += 1
        if changed:
            bp = unreal.load_asset(bp_path)
            try:
                unreal.BlueprintEditorLibrary.compile_blueprint(bp)
            except Exception:
                pass
            unreal.EditorAssetLibrary.save_asset(bp_path)
    return replaced


def cleanup_erroneous_staging() -> list[str]:
    """Remove leftover per-version import folders that are not the live mesh."""
    editor_assets = unreal.EditorAssetLibrary
    removed = []
    for path in (
        "/Game/Assets/ASV/ASVModel/v004",
        "/Game/Assets/ASV/ASVModel/v005",
        "/Game/Assets/ASV/ASVModel",
    ):
        if editor_assets.does_directory_exist(path):
            assets = list(editor_assets.list_assets(path, True, False) or [])
            # Keep a later live staging folder; only delete versioned dumps.
            if path.endswith("ASVModel") and any("/_staging" in str(item) for item in assets):
                continue
            if editor_assets.delete_directory(path):
                removed.append(path)
    return removed


def import_manifest(manifest_path: Path | None = None) -> dict:
    if manifest_path is not None and str(manifest_path).replace("\\", "/") == "__reload_main_map__":
        world = unreal.EditorLevelLibrary.get_editor_world()
        packages = []
        if world is not None:
            outer = world.get_outermost()
            if outer is not None:
                packages.append(outer)
        for extra in (
            "/Game/My_Ocean_Wave",
            "/Game/My_Water_Material",
            "/Game/Normal_Water_Inst",
            "/Game/My_Far_Ocean_Material",
            "/Game/Render_result",
        ):
            asset = unreal.EditorAssetLibrary.load_asset(extra)
            if asset is not None:
                pkg = asset.get_outermost()
                if pkg is not None and pkg not in packages:
                    packages.append(pkg)
        if packages:
            unreal.EditorLoadingAndSavingUtils.reload_packages(packages)
        loaded = unreal.EditorLoadingAndSavingUtils.load_map("/Game/Main_Map")
        return {
            "status": "reloaded",
            "ok": bool(loaded),
            "reloaded_packages": len(packages),
        }

    manifest_path = manifest_path or _manifest_path()
    manifest = load_manifest(manifest_path)
    usd_path = Path(manifest.get("usd_path") or manifest["usd_file"])
    if not usd_path.is_absolute():
        usd_path = manifest_path.parent / usd_path
    usd_path = usd_path.resolve()
    if usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"manifest does not point to a USD file: {usd_path}")
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    expected_hash = manifest.get("sha256", {}).get(usd_path.name)
    actual_hash = sha256_file(usd_path)
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError("USD SHA-256 does not match asset_manifest.json")

    target_path = (manifest.get("ue_target_mesh") or "").rstrip("/")
    if not target_path.startswith("/Game/"):
        raise RuntimeError("manifest ue_target_mesh must be a /Game path")
    destination = (manifest.get("ue_destination") or "/Game/Assets/ASV/_staging").rstrip("/")

    collision_flag = None
    if unreal.EditorAssetLibrary.does_asset_exist(target_path):
        existing = unreal.load_asset(target_path)
        if isinstance(existing, unreal.StaticMesh):
            body_setup = existing.get_editor_property("body_setup")
            if body_setup:
                collision_flag = body_setup.get_editor_property("collision_trace_flag")

    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(usd_path))
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", False)
    options = unreal.UsdStageImportOptions()
    for name, value in (
        ("import_actors", False),
        ("import_geometry", True),
        ("import_materials", True),
        ("prim_path_folder_structure", False),
        ("import_level", False),
    ):
        try:
            options.set_editor_property(name, value)
        except Exception:
            try:
                setattr(options, name, value)
            except Exception:
                pass
    try:
        task.set_editor_property("options", options)
    except Exception:
        pass

    tools = unreal.AssetToolsHelpers.get_asset_tools()
    tools.import_asset_tasks([task])
    imported = [str(path) for path in (task.get_editor_property("imported_object_paths") or [])]
    meshes = _imported_static_meshes(imported)
    if not meshes:
        raise RuntimeError(
            "UE5 imported no StaticMesh assets. Enable USD/Interchange plugins and retry."
        )

    source_mesh = meshes[0] if len(meshes) == 1 else _largest_mesh(meshes)
    overwritten = _overwrite_target(source_mesh, target_path)
    target_mesh = unreal.load_asset(overwritten)
    if not isinstance(target_mesh, unreal.StaticMesh):
        raise RuntimeError(f"overwrite did not produce a StaticMesh at {overwritten}")
    _bind_blueprints_to_mesh(target_path, target_mesh)
    _restore_collision_flag(target_mesh, collision_flag)
    material_restore = restore_slot_materials(
        overwritten, KNOWN_SLOT_MATERIALS.get(target_path, SM_ASV_SLOT_MATERIALS)
    )

    editor_assets = unreal.EditorAssetLibrary
    editor_assets.save_asset(overwritten)
    for object_path in imported:
        package = object_path.split(".", 1)[0]
        if package.startswith(destination) and editor_assets.does_asset_exist(package):
            editor_assets.save_asset(package)

    cleanup_erroneous_staging()

    result = {
        "schema_version": 2,
        "asset_name": manifest.get("asset_name"),
        "version": manifest.get("version"),
        "usd_file": str(usd_path),
        "ue_target_mesh": overwritten,
        "ue_destination": destination,
        "imported_object_paths": imported,
        "static_mesh_count": len(meshes),
        "status": "imported",
        "level_saved": False,
        "materials_restored": material_restore,
    }
    report_path = manifest_path.with_name("ue_import_result.json")
    write_json(report_path, result)
    unreal.log(f"Asset pipeline overwrote {overwritten}")
    return result


if __name__ == "__main__":
    try:
        import_manifest()
    except Exception as exc:
        unreal.log_error(f"Asset pipeline import failed: {exc}")
        raise
