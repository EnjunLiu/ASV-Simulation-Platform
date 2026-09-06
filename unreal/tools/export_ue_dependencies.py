"""Export the recursive UE Asset Registry dependency closure for Main_Map."""

import hashlib
import json
import os
from pathlib import Path

import unreal


def _package_file(package_name: str) -> Path | None:
    relative = package_name.removeprefix("/Game/").replace("/", os.sep)
    content_root = Path(unreal.Paths.project_content_dir())
    for extension in (".umap", ".uasset"):
        candidate = content_root / f"{relative}{extension}"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    output = os.environ.get("ASV_HIL_UE_OUTPUT")
    if not output:
        raise RuntimeError("ASV_HIL_UE_OUTPUT must point to the manifest output JSON")
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    root = "/Game/Main_Map"
    if not registry.get_assets_by_package_name(root):
        raise RuntimeError(f"root asset not found: {root}")
    pending = [root]
    seen: set[str] = set()
    dependencies: list[dict[str, object]] = []
    options = unreal.AssetRegistryDependencyOptions(
        include_hard_package_references=True,
        include_soft_package_references=True,
        include_searchable_names=True,
        include_soft_management_references=True,
        include_hard_management_references=True,
    )
    while pending:
        package = pending.pop()
        if package in seen or not package.startswith("/Game/"):
            continue
        seen.add(package)
        path = _package_file(package)
        if path is None:
            raise RuntimeError(f"no package file found for {package}")
        relative_file = "Content/" + package.removeprefix("/Game/") + path.suffix
        dependencies.append({
            "package": package,
            "file": relative_file,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        for dependency_name in registry.get_dependencies(package, options):
            dependency = str(dependency_name)
            if dependency.startswith("/Game/") and dependency not in seen:
                pending.append(dependency)
    document = {
        "source_project": unreal.Paths.get_project_file_path(),
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "root_asset": root,
        "dependencies": sorted(dependencies, key=lambda item: str(item["package"])),
    }
    Path(output).write_text(json.dumps(document, indent=2), encoding="utf-8")
    unreal.log(f"ASV_HIL_UE_DEPENDENCIES_EXPORTED count={len(dependencies)} output={output}")


main()
