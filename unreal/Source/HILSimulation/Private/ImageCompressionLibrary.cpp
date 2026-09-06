#include "ImageCompressionLibrary.h"
#include "Engine/TextureRenderTarget2D.h"
#include "ImageUtils.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "HAL/IConsoleManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "RenderingThread.h"
#include "RHI.h"

DEFINE_LOG_CATEGORY_STATIC(LogImageCompression, Log, All);

namespace
{
// RenderTarget readback is a synchronization point for Blueprint callers that
// invoke CaptureScene and compress in the same game-thread turn. The CVAR is
// intentionally runtime-only so the conversion can be compared without
// touching Blueprint assets or project config.
TAutoConsoleVariable<int32> CVarImageCompressionWarmupFrames(
	TEXT("edge.ImageCompression.WarmupFrames"),
	8,
	TEXT("Discard the first N render-target readbacks while UE scene resources settle."),
	ECVF_Default);

TAutoConsoleVariable<int32> CVarImageCompressionDumpFrames(
	TEXT("edge.ImageCompression.DumpFrames"),
	0,
	TEXT("Dump the first N compressed JPEG readbacks to Saved/CaptureDiagnostics for inspection."),
	ECVF_Default);

bool ReadRenderTargetPixels(
	UTextureRenderTarget2D& RenderTarget,
	TArray<FColor>& OutPixels)
{
	OutPixels.Reset();
	if (RenderTarget.SizeX <= 0 || RenderTarget.SizeY <= 0)
	{
		UE_LOG(
			LogImageCompression,
			Warning,
			TEXT("EDGE_CAPTURE_READ_INVALID size=%dx%d"),
			RenderTarget.SizeX,
			RenderTarget.SizeY);
		return false;
	}

	// SceneCapture produces tone-mapped FinalColorLDR.  The authored target is
	// RTF_RGBA8, so convert the linear readback once to the display sRGB values
	// shown by the Render Target editor before JPEG compression.
	// Flush a SceneCapture render queued earlier in this game-thread turn before
	// the CPU readback.
	FlushRenderingCommands();

	FRenderTarget* RTResource = RenderTarget.GameThread_GetRenderTargetResource();
	if (RTResource == nullptr)
	{
		// A newly created target may not have a resource yet.  Initialize it
		// without changing any authored color-space settings, then retry once.
		RenderTarget.UpdateResourceImmediate(false);
		FlushRenderingCommands();
		RTResource = RenderTarget.GameThread_GetRenderTargetResource();
	}
	if (RTResource == nullptr)
	{
		UE_LOG(
			LogImageCompression,
			Warning,
			TEXT("EDGE_CAPTURE_READ_NO_RESOURCE size=%dx%d"),
			RenderTarget.SizeX,
			RenderTarget.SizeY);
		return false;
	}

	FReadSurfaceDataFlags ReadFlags(RCM_UNorm);
	ReadFlags.SetLinearToGamma(false);
	if (!RTResource->ReadPixels(OutPixels, ReadFlags))
	{
		UE_LOG(
			LogImageCompression,
			Warning,
			TEXT("EDGE_CAPTURE_READ_FAILED size=%dx%d"),
			RenderTarget.SizeX,
			RenderTarget.SizeY);
		OutPixels.Reset();
		return false;
	}
	const int32 ExpectedPixels = RenderTarget.SizeX * RenderTarget.SizeY;
	if (OutPixels.Num() != ExpectedPixels)
	{
		UE_LOG(
			LogImageCompression,
			Warning,
			TEXT("EDGE_CAPTURE_READ_SIZE_MISMATCH expected=%d actual=%d"),
			ExpectedPixels,
			OutPixels.Num());
		OutPixels.Reset();
		return false;
	}

	for (FColor& Pixel : OutPixels)
	{
		const FLinearColor Linear(
			static_cast<float>(Pixel.R) / 255.0f,
			static_cast<float>(Pixel.G) / 255.0f,
			static_cast<float>(Pixel.B) / 255.0f,
			static_cast<float>(Pixel.A) / 255.0f);
		Pixel = Linear.ToFColor(true);
	}

	uint8 MaxChannel = 0;
	uint64 ChannelSums[3] = {0, 0, 0};
	for (const FColor& Pixel : OutPixels)
	{
		MaxChannel = FMath::Max(MaxChannel, FMath::Max3(Pixel.R, Pixel.G, Pixel.B));
		ChannelSums[0] += Pixel.R;
		ChannelSums[1] += Pixel.G;
		ChannelSums[2] += Pixel.B;
	}
	static int32 CaptureStatsLogCount = 0;
	++CaptureStatsLogCount;
	if (CaptureStatsLogCount <= 5 || CaptureStatsLogCount % 250 == 0)
	{
		const double PixelCount = static_cast<double>(OutPixels.Num());
		UE_LOG(
			LogImageCompression,
			Display,
			TEXT("EDGE_CAPTURE_STATS count=%d size=%dx%d srgb=%d mean_rgb=(%.1f,%.1f,%.1f) max=%d manual_srgb_conversion=1"),
			CaptureStatsLogCount,
			RenderTarget.SizeX,
			RenderTarget.SizeY,
			RenderTarget.SRGB ? 1 : 0,
			static_cast<double>(ChannelSums[0]) / PixelCount,
			static_cast<double>(ChannelSums[1]) / PixelCount,
			static_cast<double>(ChannelSums[2]) / PixelCount,
			MaxChannel);
	}

	const int32 WarmupFrames = FMath::Max(
		0,
		CVarImageCompressionWarmupFrames.GetValueOnGameThread());
	if (CaptureStatsLogCount <= WarmupFrames)
	{
		UE_LOG(
			LogImageCompression,
			Display,
			TEXT("EDGE_CAPTURE_WARMUP_SKIP count=%d/%d size=%dx%d mean_rgb=(%.1f,%.1f,%.1f)"),
			CaptureStatsLogCount,
			WarmupFrames,
			RenderTarget.SizeX,
			RenderTarget.SizeY,
			static_cast<double>(ChannelSums[0]) / static_cast<double>(OutPixels.Num()),
			static_cast<double>(ChannelSums[1]) / static_cast<double>(OutPixels.Num()),
			static_cast<double>(ChannelSums[2]) / static_cast<double>(OutPixels.Num()));
		OutPixels.Reset();
		return false;
	}
	if (MaxChannel == 0)
	{
		UE_LOG(
			LogImageCompression,
			Warning,
			TEXT("EDGE_CAPTURE_READ_BLACK size=%dx%d; capture may not have completed or the scene/clear color is black"),
			RenderTarget.SizeX,
			RenderTarget.SizeY);
	}
	else
	{
		UE_LOG(
			LogImageCompression,
			VeryVerbose,
			TEXT("EDGE_CAPTURE_READ_OK size=%dx%d max_channel=%d manual_srgb_conversion=1"),
			RenderTarget.SizeX,
			RenderTarget.SizeY,
			MaxChannel);
	}
	return true;
}
} // namespace

void UImageCompressionLibrary::CompressRenderTargetToPNG(UTextureRenderTarget2D* RenderTarget, TArray<uint8>& OutPNGData)
{
	OutPNGData.Empty();
	if (!RenderTarget) return;

	TArray<FColor> Pixels;
	if (ReadRenderTargetPixels(*RenderTarget, Pixels))
	{
		TArray64<uint8> Compressed;
		FImageUtils::PNGCompressImageArray(
			RenderTarget->SizeX, RenderTarget->SizeY, Pixels, Compressed);
		OutPNGData.Reset(IntCastChecked<int32>(Compressed.Num()));
		OutPNGData.Append(Compressed.GetData(), IntCastChecked<int32>(Compressed.Num()));
	}
}

void UImageCompressionLibrary::CompressRenderTargetToJPEG(UTextureRenderTarget2D* RenderTarget, TArray<uint8>& OutJPEGData, int32 Quality)
{
	OutJPEGData.Empty();
	if (!RenderTarget) return;

	TArray<FColor> Pixels;
	if (ReadRenderTargetPixels(*RenderTarget, Pixels))
	{
		CompressPixelArrayToJPEG(RenderTarget->SizeX, RenderTarget->SizeY, Pixels, OutJPEGData, Quality);
		static int32 DumpedFrameCount = 0;
		const int32 DumpFrameLimit = FMath::Max(
			0,
			CVarImageCompressionDumpFrames.GetValueOnGameThread());
		if (DumpFrameLimit > DumpedFrameCount && OutJPEGData.Num() > 0)
		{
			const FString Directory = FPaths::Combine(
				FPaths::ProjectSavedDir(), TEXT("CaptureDiagnostics"));
			IFileManager::Get().MakeDirectory(*Directory, true);
			const FString Path = FPaths::Combine(
				Directory,
				FString::Printf(TEXT("capture_%04d.jpg"), DumpedFrameCount + 1));
			if (FFileHelper::SaveArrayToFile(OutJPEGData, *Path))
			{
				++DumpedFrameCount;
				UE_LOG(
					LogImageCompression,
					Display,
					TEXT("EDGE_CAPTURE_DUMP path=%s count=%d/%d bytes=%d"),
					*Path,
					DumpedFrameCount,
					DumpFrameLimit,
					OutJPEGData.Num());
			}
		}
	}
}

void UImageCompressionLibrary::CompressPixelArrayToPNG(int32 Width, int32 Height, const TArray<FColor>& PixelData, TArray<uint8>& OutPNGData)
{
	OutPNGData.Empty();
	if (PixelData.Num() != Width * Height || Width <= 0 || Height <= 0) return;
	TArray64<uint8> Compressed;
	FImageUtils::PNGCompressImageArray(Width, Height, PixelData, Compressed);
	OutPNGData.Reset(IntCastChecked<int32>(Compressed.Num()));
	OutPNGData.Append(Compressed.GetData(), IntCastChecked<int32>(Compressed.Num()));
}

void UImageCompressionLibrary::CompressPixelArrayToJPEG(int32 Width, int32 Height, const TArray<FColor>& PixelData, TArray<uint8>& OutJPEGData, int32 Quality)
{
	OutJPEGData.Empty();
	if (PixelData.Num() != Width * Height || Width <= 0 || Height <= 0) return;

	Quality = FMath::Clamp(Quality, 1, 100);

	IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
	TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::JPEG);

	if (ImageWrapper.IsValid() && ImageWrapper->SetRaw(
		(const uint8*)PixelData.GetData(),
		PixelData.Num() * sizeof(FColor),
		Width, Height,
		ERGBFormat::BGRA, 8))
	{
		const TArray64<uint8>& Compressed = ImageWrapper->GetCompressed(Quality);
		OutJPEGData = TArray<uint8>(Compressed);   // 安全转换
	}
}
