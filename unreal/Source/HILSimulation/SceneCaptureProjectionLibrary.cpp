#include "SceneCaptureProjectionLibrary.h"

#include "Camera/CameraTypes.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Kismet/GameplayStatics.h"
#include "SceneView.h"

namespace
{
	bool GetCaptureViewProjection(
		USceneCaptureComponent2D* SceneCapture,
		FIntPoint& OutTargetSize,
		FMatrix& OutViewProjectionMatrix)
	{
		if (!IsValid(SceneCapture) || !IsValid(SceneCapture->TextureTarget))
		{
			return false;
		}

		OutTargetSize = FIntPoint(
			SceneCapture->TextureTarget->SizeX,
			SceneCapture->TextureTarget->SizeY);

		if (OutTargetSize.X <= 0 || OutTargetSize.Y <= 0)
		{
			return false;
		}

		FMinimalViewInfo ViewInfo;
		SceneCapture->GetCameraView(0.0f, ViewInfo);

		// GetCameraView does not copy this SceneCapture-specific override.
		// Supplying it here keeps the generated projection in step with the
		// renderer when no custom projection matrix is active.
		if (!SceneCapture->bUseCustomProjectionMatrix &&
			SceneCapture->bOverride_CustomNearClippingPlane)
		{
			ViewInfo.PerspectiveNearClipPlane =
				SceneCapture->CustomNearClippingPlane;
		}

		TOptional<FMatrix> CustomProjectionMatrix;
		if (SceneCapture->bUseCustomProjectionMatrix)
		{
			CustomProjectionMatrix = SceneCapture->CustomProjectionMatrix;
		}

		FMatrix ViewMatrix;
		FMatrix ProjectionMatrix;
		UGameplayStatics::CalculateViewProjectionMatricesFromMinimalView(
			ViewInfo,
			CustomProjectionMatrix,
			ViewMatrix,
			ProjectionMatrix,
			OutViewProjectionMatrix);

		return !OutViewProjectionMatrix.ContainsNaN();
	}

	bool ProjectWithPreparedView(
		USceneCaptureComponent2D* SceneCapture,
		const FIntPoint& TargetSize,
		const FMatrix& ViewProjectionMatrix,
		const FVector& WorldPosition,
		FVector2D& OutPixelPosition)
	{
		OutPixelPosition = FVector2D::ZeroVector;

		if (WorldPosition.ContainsNaN())
		{
			return false;
		}

		// SceneCaptureComponent2D follows UE coordinates: +X forward,
		// +Y right, +Z up. Rejecting non-positive forward depth prevents
		// points behind the camera from being mirrored into the image.
		const FVector CaptureSpacePosition =
			SceneCapture->GetComponentTransform().InverseTransformPosition(
				WorldPosition);
		if (!FMath::IsFinite(CaptureSpacePosition.X) ||
			CaptureSpacePosition.X <= KINDA_SMALL_NUMBER)
		{
			return false;
		}

		const FIntRect RenderRect(FIntPoint::ZeroValue, TargetSize);
		const bool bProjected = FSceneView::ProjectWorldToScreen(
			WorldPosition,
			RenderRect,
			ViewProjectionMatrix,
			OutPixelPosition,
			true);

		return bProjected &&
			FMath::IsFinite(OutPixelPosition.X) &&
			FMath::IsFinite(OutPixelPosition.Y);
	}

	bool GetComponentLocalBounds(
		UPrimitiveComponent* TargetComponent,
		FVector& OutLocalMin,
		FVector& OutLocalMax)
	{
		OutLocalMin = FVector::ZeroVector;
		OutLocalMax = FVector::ZeroVector;

		if (!IsValid(TargetComponent))
		{
			return false;
		}

		if (const UStaticMeshComponent* StaticMeshComponent =
			Cast<UStaticMeshComponent>(TargetComponent))
		{
			if (!IsValid(StaticMeshComponent->GetStaticMesh()))
			{
				return false;
			}

			StaticMeshComponent->GetLocalBounds(OutLocalMin, OutLocalMax);
		}
		else
		{
			const FBoxSphereBounds LocalBounds =
				TargetComponent->CalcBounds(FTransform::Identity);
			OutLocalMin = LocalBounds.Origin - LocalBounds.BoxExtent;
			OutLocalMax = LocalBounds.Origin + LocalBounds.BoxExtent;
		}

		if (OutLocalMin.ContainsNaN() || OutLocalMax.ContainsNaN())
		{
			return false;
		}

		return OutLocalMin.X <= OutLocalMax.X &&
			OutLocalMin.Y <= OutLocalMax.Y &&
			OutLocalMin.Z <= OutLocalMax.Z;
	}
}

bool USceneCaptureProjectionLibrary::ProjectWorldToSceneCapturePixels(
	USceneCaptureComponent2D* SceneCapture,
	FVector WorldPosition,
	FVector2D& PixelPosition)
{
	PixelPosition = FVector2D::ZeroVector;

	FIntPoint TargetSize;
	FMatrix ViewProjectionMatrix;
	if (!GetCaptureViewProjection(
		SceneCapture,
		TargetSize,
		ViewProjectionMatrix))
	{
		return false;
	}

	return ProjectWithPreparedView(
		SceneCapture,
		TargetSize,
		ViewProjectionMatrix,
		WorldPosition,
		PixelPosition);
}

bool USceneCaptureProjectionLibrary::GetComponentBBoxOnSceneCapture(
	UPrimitiveComponent* TargetComponent,
	USceneCaptureComponent2D* SceneCapture,
	TArray<int32>& BBox)
{
	BBox.Reset();

	if (!IsValid(TargetComponent))
	{
		return false;
	}

	FIntPoint TargetSize;
	FMatrix ViewProjectionMatrix;
	if (!GetCaptureViewProjection(
		SceneCapture,
		TargetSize,
		ViewProjectionMatrix))
	{
		return false;
	}

	FVector LocalMin;
	FVector LocalMax;
	if (!GetComponentLocalBounds(TargetComponent, LocalMin, LocalMax))
	{
		return false;
	}

	const FVector LocalCorners[8] =
	{
		FVector(LocalMin.X, LocalMin.Y, LocalMin.Z),
		FVector(LocalMax.X, LocalMin.Y, LocalMin.Z),
		FVector(LocalMin.X, LocalMax.Y, LocalMin.Z),
		FVector(LocalMax.X, LocalMax.Y, LocalMin.Z),
		FVector(LocalMin.X, LocalMin.Y, LocalMax.Z),
		FVector(LocalMax.X, LocalMin.Y, LocalMax.Z),
		FVector(LocalMin.X, LocalMax.Y, LocalMax.Z),
		FVector(LocalMax.X, LocalMax.Y, LocalMax.Z)
	};

	const FTransform ComponentTransform =
		TargetComponent->GetComponentTransform();

	double RawMinX = TNumericLimits<double>::Max();
	double RawMinY = TNumericLimits<double>::Max();
	double RawMaxX = TNumericLimits<double>::Lowest();
	double RawMaxY = TNumericLimits<double>::Lowest();

	for (const FVector& LocalCorner : LocalCorners)
	{
		const FVector WorldCorner =
			ComponentTransform.TransformPosition(LocalCorner);

		FVector2D PixelPosition;
		if (!ProjectWithPreparedView(
			SceneCapture,
			TargetSize,
			ViewProjectionMatrix,
			WorldCorner,
			PixelPosition))
		{
			// Do not emit a misleading box when any part of the local bounds
			// reaches or crosses behind the capture plane.
			return false;
		}

		RawMinX = FMath::Min(RawMinX, PixelPosition.X);
		RawMinY = FMath::Min(RawMinY, PixelPosition.Y);
		RawMaxX = FMath::Max(RawMaxX, PixelPosition.X);
		RawMaxY = FMath::Max(RawMaxY, PixelPosition.Y);
	}

	const double LastPixelX = static_cast<double>(TargetSize.X - 1);
	const double LastPixelY = static_cast<double>(TargetSize.Y - 1);

	if (RawMaxX < 0.0 || RawMinX > LastPixelX ||
		RawMaxY < 0.0 || RawMinY > LastPixelY)
	{
		return false;
	}

	// Clamp in floating point before integer conversion so extreme off-screen
	// projections cannot overflow FloorToInt/CeilToInt.
	const int32 MinX = FMath::FloorToInt(
		FMath::Clamp(RawMinX, 0.0, LastPixelX));
	const int32 MinY = FMath::FloorToInt(
		FMath::Clamp(RawMinY, 0.0, LastPixelY));
	const int32 MaxX = FMath::CeilToInt(
		FMath::Clamp(RawMaxX, 0.0, LastPixelX));
	const int32 MaxY = FMath::CeilToInt(
		FMath::Clamp(RawMaxY, 0.0, LastPixelY));

	if (MaxX <= MinX || MaxY <= MinY)
	{
		return false;
	}

	BBox.Reserve(4);
	BBox.Add(MinX);
	BBox.Add(MinY);
	BBox.Add(MaxX);
	BBox.Add(MaxY);
	return true;
}
