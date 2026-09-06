"""Write red/blue/white/gray ASV payloads. Hull + both base plates share one material."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "../ASVModel.usd"

# Linear sRGB, close to UE M_HullRed / M_HullBlue / M_HullWhite / M_HullGray.
HULL_COLORS = {
    "red": (0.72, 0.05, 0.04),
    "blue": (0.05, 0.18, 0.62),
    "white": (0.82, 0.84, 0.85),
    "gray": (0.42, 0.44, 0.47),
}

_SHADER = """
            def Shader "Principled_PBR"
            {{
                uniform token info:id = "UsdPreviewSurface"
                float inputs:clearcoat = 0
                float inputs:clearcoatRoughness = 0.45
                color3f inputs:diffuseColor = ({r}, {g}, {b})
                float inputs:ior = 1.45
                float inputs:metallic = 0
                float inputs:opacity = 1
                float inputs:roughness = 0.62
                float inputs:specular = 0.5
                token outputs:surface
            }}
"""


def usda_for(color: str, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    shader = _SHADER.format(r=r, g=g, b=b)
    return f"""#usda 1.0
(
    defaultPrim = "ASV"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @{SOURCE}@
    ]
)

over "ASV"
{{
    over "ASV_Root"
    {{
        over "Hull"
        {{
            over "Hull_Mesh" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {{
                rel material:binding = </ASV/_materials/Hull_Color>
            }}
        }}
        over "Left_Base_Plate"
        {{
            over "Left_Base_Plate_Mesh" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {{
                rel material:binding = </ASV/_materials/Hull_Color>
            }}
        }}
        over "Right_Base_Plate"
        {{
            over "Right_Base_Plate_Mesh" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {{
                rel material:binding = </ASV/_materials/Hull_Color>
            }}
        }}
    }}

    over "_materials"
    {{
        def Material "Hull_Color" (
            prepend apiSchemas = ["ColorSpaceAPI"]
        )
        {{
            uniform token colorSpace:name = "lin_rec709_scene"
            token outputs:surface.connect = </ASV/_materials/Hull_Color/Principled_PBR.outputs:surface>
{shader}
        }}
    }}
}}
"""


def write_all(out_dir: Path | None = None) -> list[Path]:
    dest = Path(out_dir) if out_dir is not None else ROOT
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, rgb in HULL_COLORS.items():
        path = dest / f"ASVModel_{name}.usda"
        path.write_text(usda_for(name, rgb), encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write_all():
        print("wrote", path)
