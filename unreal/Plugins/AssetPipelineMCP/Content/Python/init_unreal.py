"""Plugin Python startup: register the asset pipeline MCP toolset."""

import unreal
from toolset_registry.registration import Registration

try:
    import asset_pipeline_toolset

    if Registration([asset_pipeline_toolset.AssetPipelineTools]).register():
        unreal.log("Registered AssetPipelineTools plugin with ToolsetRegistry")
    else:
        unreal.log_warning("ToolsetRegistry not available; AssetPipelineTools plugin not registered")
except Exception as exc:
    unreal.log_warning(f"AssetPipelineTools plugin failed to load: {exc}")
