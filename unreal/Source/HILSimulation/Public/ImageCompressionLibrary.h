#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/TextureRenderTarget2D.h"
#include "ImageCompressionLibrary.generated.h"

UCLASS()
class HILSIMULATION_API UImageCompressionLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** 把 Render Target 直接压缩成 PNG 字节数组（无损，文件较大） */
	UFUNCTION(BlueprintCallable, Category = "Image Compression")
	static void CompressRenderTargetToPNG(UTextureRenderTarget2D* RenderTarget, TArray<uint8>& OutPNGData);

	/** 把 Render Target 直接压缩成 JPEG 字节数组（有损，压缩率高、速度快） */
	UFUNCTION(BlueprintCallable, Category = "Image Compression")
	static void CompressRenderTargetToJPEG(UTextureRenderTarget2D* RenderTarget, TArray<uint8>& OutJPEGData, int32 Quality = 95);

	/** 如果你已经有像素数组，可直接压缩成 PNG */
	UFUNCTION(BlueprintCallable, Category = "Image Compression")
	static void CompressPixelArrayToPNG(int32 Width, int32 Height, const TArray<FColor>& PixelData, TArray<uint8>& OutPNGData);

	/** 如果你已经有像素数组，可直接压缩成 JPEG */
	UFUNCTION(BlueprintCallable, Category = "Image Compression")
	static void CompressPixelArrayToJPEG(int32 Width, int32 Height, const TArray<FColor>& PixelData, TArray<uint8>& OutJPEGData, int32 Quality = 95);
};
