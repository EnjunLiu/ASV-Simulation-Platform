using UnrealBuildTool;
using System.Collections.Generic;

public class HILPlatformEditorTarget : TargetRules
{
	public HILPlatformEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V7;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_8;

		ExtraModuleNames.AddRange( new string[] { "HILSimulation" } );
	}
}
