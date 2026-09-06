#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "SceneCaptureProjectionLibrary.generated.h"

class UPrimitiveComponent;
class USceneCaptureComponent2D;

/**
 * Projection helpers for mapping world-space geometry to a SceneCapture2D
 * render target. Pixel coordinates use a top-left origin.
 */
UCLASS()
class HILSIMULATION_API USceneCaptureProjectionLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Projects a world-space point into the SceneCapture2D TextureTarget.
	 *
	 * Returns false when the capture/target is invalid, the target has no
	 * resolution, the point is at or behind the capture plane, or projection
	 * cannot produce finite pixel coordinates. Off-screen points in front of
	 * the camera still return true and can produce coordinates outside the
	 * render target.
	 */
	UFUNCTION(
		BlueprintPure,
		Category = "HIL|Vision|Scene Capture",
		meta = (DisplayName = "Project World to Scene Capture Pixels")
	)
	static bool ProjectWorldToSceneCapturePixels(
		USceneCaptureComponent2D* SceneCapture,
		FVector WorldPosition,
		FVector2D& PixelPosition
	);

	/**
	 * Projects the eight corners of a component-local 3D bounds into the
	 * SceneCapture2D TextureTarget and returns the clipped 2D bounding box.
	 *
	 * StaticMeshComponent uses its mesh-local bounds. Other primitive
	 * components fall back to CalcBounds with an identity transform.
	 *
	 * BBox contains exactly [MinX, MinY, MaxX, MaxY] when true and is empty
	 * when false. A box that crosses the camera plane is rejected rather than
	 * producing a mirrored or unbounded result.
	 */
	UFUNCTION(
		BlueprintPure,
		Category = "HIL|Vision|Scene Capture",
		meta = (DisplayName = "Get Component BBox on Scene Capture")
	)
	static bool GetComponentBBoxOnSceneCapture(
		UPrimitiveComponent* TargetComponent,
		USceneCaptureComponent2D* SceneCapture,
		TArray<int32>& BBox
	);
};
