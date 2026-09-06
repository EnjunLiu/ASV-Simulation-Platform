"""MCP-visible tools for the Blender -> USD -> UE mesh overwrite pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import unreal
import toolset_registry

TOOLS_DIR = Path(unreal.Paths.project_dir()) / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


@unreal.uclass()
class AssetPipelineTools(unreal.ToolsetDefinition):
    """Import a published USD onto the StaticMesh named in its manifest."""

    @toolset_registry.tool_call
    @staticmethod
    def import_usd_manifest(manifest_path: str) -> str:
        """Overwrite ue_target_mesh from a published asset_manifest.json.

        Args:
            manifest_path: Absolute path to asset_manifest.json.

        Returns:
            JSON object with status, ue_target_mesh, and imported paths.
        """
        from importlib import reload
        import import_usd_manifest

        reload(import_usd_manifest)
        result = import_usd_manifest.import_manifest(Path(manifest_path))
        return json.dumps(result, ensure_ascii=False)

    @toolset_registry.tool_call
    @staticmethod
    def restore_asv_materials() -> str:
        """Rebind SM_ASV and SM_Target slots to English materials. Does not save Main_Map."""
        from import_usd_manifest import restore_slot_materials

        result = {
            "SM_ASV": restore_slot_materials("/Game/ASVModel/SM_ASV"),
            "SM_Target": restore_slot_materials("/Game/ASVModel/SM_Target"),
        }
        return json.dumps(result, ensure_ascii=False)

    @toolset_registry.tool_call
    @staticmethod
    def asset_exists(object_path: str) -> str:
        """Return whether a Content asset path exists.

        Args:
            object_path: Unreal object path such as /Game/BP_ASV or /Game/Main_Map.

        Returns:
            JSON with exists=true/false.
        """
        exists = bool(unreal.EditorAssetLibrary.does_asset_exist(object_path))
        return json.dumps({"object_path": object_path, "exists": exists})
