"""Visual Gerstner ocean mesh. Warp GPU -> Fabric, Tessendorf FFT, MDL water."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np

from .fft_ocean import TessendorfField
from .waves import WaveField

WATER_MAT = "/Visual_materials/water"
DEFAULT_UV_TILE_M = 4.0
# PreviewSurface at grazing (ship camera) with 0.08 is a sheet of glass.
# 0.20 + chop normals break the long hull streaks without turning the sea into plastic.
WATER_ROUGHNESS = 0.20
WATER_SPECULAR = (0.04, 0.045, 0.05)
OCEAN_SIZE_M = 40.0
OCEAN_GRID_N = 513
FAR_OCEAN_SIZE_M = 400.0
FAR_OCEAN_GRID_N = 257
FAR_OCEAN_INNER_M = OCEAN_SIZE_M
FAR_OCEAN_Z_BIAS = -0.003
FAR_OCEAN_MIN_LAM_M = 8.0
CHOP_MAP_N = 4096
CHOP_LAM_MAX_M = 0.55
CHOP_LAM_MIN_M = 0.02
CHOP_WIND = (1.0, 0.18)
CHOP_WIND_SPEED = 7.0
CHOP_SLOPE_RMS = 0.45
CHOP_SCROLL_MPS = (0.0, 0.0)
CHOP_FFT_N = 256
WATER_DIFFUSE = (0.002, 0.02, 0.04)
_WATER_SCROLL = None
_WATER_FILE = None
_CHOP_FFT = None
_CHOP_DYNAMIC = None
_CHOP_DYNAMIC_NAME = "isaac_hydro_chop"
_CHOP_T = None


def make_rest_grid(size: float = OCEAN_SIZE_M, n: int = OCEAN_GRID_N) -> tuple[np.ndarray, list[int], list[int]]:
    n = int(n)
    xs = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float32)
    ys = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    rest = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n, dtype=np.float32)], axis=1)
    i = np.arange(n - 1, dtype=np.int32)
    j = np.arange(n - 1, dtype=np.int32)
    ii, jj = np.meshgrid(i, j, indexing="xy")
    a = (jj * n + ii).ravel()
    indices = np.stack([a, a + 1, a + n + 1, a + n], axis=1).ravel().tolist()
    counts = np.full((n - 1) * (n - 1), 4, dtype=np.int32).tolist()
    return rest, counts, indices


def make_ring_grid(inner: float, outer: float, n: int) -> tuple[np.ndarray, list[int], list[int]]:
    """Square annulus: outer disc minus inner hole. Same Gerstner rest as the near patch."""
    rest, _, _ = make_rest_grid(outer, n)
    n = int(n)
    i = np.arange(n - 1, dtype=np.int32)
    j = np.arange(n - 1, dtype=np.int32)
    ii, jj = np.meshgrid(i, j, indexing="xy")
    a = (jj * n + ii).ravel()
    quads = np.stack([a, a + 1, a + n + 1, a + n], axis=1)
    pts = rest[quads]
    cheb = np.maximum(np.abs(pts[..., 0]), np.abs(pts[..., 1]))
    keep = np.any(cheb >= 0.5 * float(inner) - 1e-4, axis=1)
    quads = quads[keep]
    indices = quads.ravel().tolist()
    counts = np.full(int(quads.shape[0]), 4, dtype=np.int32).tolist()
    return rest, counts, indices


def _set_vec3f(attr, arr, Gf, Vt):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
    try:
        attr.Set(Vt.Vec3fArray.FromNumpy(arr))
        return
    except Exception:
        pass
    attr.Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in arr])


def _set_vec2f(attr, arr, Gf, Vt):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
    try:
        attr.Set(Vt.Vec2fArray.FromNumpy(arr))
        return
    except Exception:
        pass
    attr.Set([Gf.Vec2f(float(p[0]), float(p[1])) for p in arr])


def _assets_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))


def default_normal_tex() -> str:
    chop = os.path.join(_assets_dir(), "water_chop_normal.png")
    if os.path.isfile(chop):
        return chop
    return os.path.join(_assets_dir(), "water_tiling_normal.png")


def snap_origin(xy, size: float = OCEAN_SIZE_M, n: int = OCEAN_GRID_N) -> np.ndarray:
    """Quantize XY to the near-grid spacing so vertices do not swim."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
    dx = float(size) / float(max(int(n) - 1, 1))
    return np.round(xy / dx) * dx


def camera_xy_world() -> np.ndarray | None:
    try:
        import omni.kit.viewport.utility as vut
        import omni.usd
        from pxr import Usd, UsdGeom

        vp = vut.get_active_viewport()
        if vp is None:
            return None
        path = getattr(vp, "camera_path", None) or vp.get_active_camera()
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(str(path))
        if not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array([float(xf[3][0]), float(xf[3][1])], dtype=np.float64)
    except Exception:
        return None


def _chop_fft() -> TessendorfField:
    global _CHOP_FFT
    if _CHOP_FFT is None:
        _CHOP_FFT = TessendorfField(
            n=CHOP_FFT_N,
            tile_m=DEFAULT_UV_TILE_M,
            lam_min=CHOP_LAM_MIN_M,
            lam_max=CHOP_LAM_MAX_M,
            wind=CHOP_WIND,
            wind_speed=CHOP_WIND_SPEED,
            height_rms=0.012,
            choppiness=0.0,
            seed=11,
        )
        # Match the old baker: scale by slope, not height.
        rms = float(np.sqrt(np.mean(_CHOP_FFT.sx * _CHOP_FFT.sx + _CHOP_FFT.sy * _CHOP_FFT.sy)))
        if rms > 1e-12:
            scale = float(CHOP_SLOPE_RMS) / rms
            _CHOP_FFT._h0 *= scale
            _CHOP_FFT._h0_conj_neg *= scale
            _CHOP_FFT.height_rms *= scale
            _CHOP_FFT._t = None
            _CHOP_FFT.evolve(0.0)
    return _CHOP_FFT


def _upload_chop_dynamic(rgb_u8: np.ndarray) -> bool:
    """Upload HxWx3 uint8 OpenGL normals as RGBA8. Never pass numpy to set_bytes_data."""
    global _CHOP_DYNAMIC
    if _CHOP_DYNAMIC is False:
        return False
    n = int(rgb_u8.shape[0])
    rgba = np.empty((n, n, 4), dtype=np.uint8)
    rgba[..., :3] = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
    rgba[..., 3] = 255
    rgba = np.ascontiguousarray(rgba)
    try:
        import omni.ui as ui

        if _CHOP_DYNAMIC is None:
            _CHOP_DYNAMIC = ui.DynamicTextureProvider(_CHOP_DYNAMIC_NAME)
        if hasattr(_CHOP_DYNAMIC, "set_data_array"):
            _CHOP_DYNAMIC.set_data_array(rgba, [n, n])
            return True
        pixels = rgba.reshape(-1).tolist()
        try:
            import omni.gpu_foundation_factory as gf

            _CHOP_DYNAMIC.set_bytes_data(pixels, [n, n], gf.TextureFormat.RGBA8_UNORM)
        except TypeError:
            _CHOP_DYNAMIC.set_bytes_data(pixels, [n, n])
        return True
    except Exception as exc:
        print("dynamic chop tex skip:", exc)
        _CHOP_DYNAMIC = False
        return False


def phillips_chop_height(
    n: int,
    tile_m: float,
    *,
    lam_max: float = CHOP_LAM_MAX_M,
    lam_min: float = CHOP_LAM_MIN_M,
    wind: tuple[float, float] = CHOP_WIND,
    wind_speed: float = CHOP_WIND_SPEED,
    slope_rms: float = CHOP_SLOPE_RMS,
    seed: int = 7,
) -> np.ndarray:
    """Tileable Phillips-spectrum height (Tessendorf snapshot). λ below vertex Gerstner."""
    n = int(n)
    L = float(tile_m)
    dx = L / float(n)
    lam_min = max(float(lam_min), 5.0 * dx)
    lam_max = min(float(lam_max), 0.55 * L)
    if lam_min >= lam_max:
        lam_min = 0.4 * lam_max
    freq = np.fft.fftfreq(n, d=dx) * (2.0 * np.pi)
    kx, ky = np.meshgrid(freq, freq, indexing="xy")
    k2 = kx * kx + ky * ky
    k = np.sqrt(k2)
    wd = np.array([float(wind[0]), float(wind[1])], dtype=np.float64)
    wn = float(np.linalg.norm(wd))
    wd = wd / wn if wn > 1e-12 else np.array([1.0, 0.0])
    khx = np.divide(kx, k, out=np.zeros_like(kx), where=k > 1e-12)
    khy = np.divide(ky, k, out=np.zeros_like(ky), where=k > 1e-12)
    mu = khx * wd[0] + khy * wd[1]
    spread = mu * mu + 0.18
    Lw = max(float(wind_speed) ** 2 / 9.81, 1e-3)
    l_small = 0.5 * lam_min
    k_lo = 2.0 * np.pi / lam_max
    k_hi = 2.0 * np.pi / lam_min
    valid = (k >= k_lo) & (k <= k_hi)
    P = np.zeros_like(k2)
    P[valid] = (
        np.exp(-1.0 / np.maximum(k2[valid] * Lw * Lw, 1e-12))
        / np.maximum(k2[valid] * k2[valid], 1e-18)
        * np.exp(-k2[valid] * l_small * l_small)
        * spread[valid]
    )
    P[0, 0] = 0.0
    rng = np.random.default_rng(int(seed))
    amp = np.sqrt(np.maximum(P, 0.0) * 0.5)
    spec = amp * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    spec[0, 0] = 0.0
    z = np.fft.ifft2(spec).real.astype(np.float64)
    ddx, ddy = _height_slope(z, L)
    rms = float(np.sqrt(np.mean(ddx * ddx + ddy * ddy)))
    if rms > 1e-12:
        z *= float(slope_rms) / rms
    return z


def _height_slope(z: np.ndarray, tile_m: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(z.shape[0])
    s = n / (2.0 * float(tile_m))
    dx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) * s
    dy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) * s
    return dx, dy


def _write_png_rgb(path: str, rgb_u8: np.ndarray) -> str:
    import struct
    import zlib

    n = int(rgb_u8.shape[0])
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + rgb_u8[i].tobytes() for i in range(n))
    ihdr = struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)
    return path


def bake_chop_normal(path: str | None = None, n: int = CHOP_MAP_N, tile_m: float = DEFAULT_UV_TILE_M) -> str:
    """Tileable OpenGL normal map: Phillips chop, λ below the vertex Gerstner band."""
    n = int(n)
    tile_m = float(tile_m)
    z = phillips_chop_height(n, tile_m)
    dx, dy = _height_slope(z, tile_m)
    nx = -dx
    ny = -dy
    nz = np.ones_like(z)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([nx * inv, ny * inv, nz * inv], axis=2)
    rgb = np.clip(rgb * 0.5 + 0.5, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    out = path or os.path.join(_assets_dir(), "water_chop_normal.png")
    return _write_png_rgb(out, rgb_u8)


def _connect_chop_normal(stage, path: str, shader, normal_input, tex: str):
    """UsdUVTexture + world-lock Transform2d. tex may be a file or dynamic:// URI."""
    from pxr import Sdf, UsdShade

    global _WATER_SCROLL, _WATER_FILE
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    uv_out = st_out
    try:
        xf = UsdShade.Shader.Define(stage, path + "/stXform")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((1.0, 1.0))
        xf.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.0)
        trans = xf.CreateInput("translation", Sdf.ValueTypeNames.Float2)
        trans.Set((0.0, 0.0))
        uv_out = xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        _WATER_SCROLL = trans
        print("water chop UV world-lock UsdTransform2d")
    except Exception as exc:
        print("UsdTransform2d skip:", exc)
    ntex = UsdShade.Shader.Define(stage, path + "/NormalTex")
    ntex.CreateIdAttr("UsdUVTexture")
    file_in = ntex.CreateInput("file", Sdf.ValueTypeNames.Asset)
    file_in.Set(tex)
    _WATER_FILE = file_in
    ntex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
    ntex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    ntex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    try:
        ntex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    except Exception:
        pass
    # Packed OpenGL normals [0,1] -> tangent space [-1,1].
    try:
        ntex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
        ntex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
    except Exception:
        pass
    rgb = ntex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    normal_input.ConnectToSource(rgb)
    print("water tiling normal", tex)


def _make_omni_surface(stage, path: str, tex: str | None) -> bool:
    """Not used at runtime. Isaac 6 RTX hid the mesh: OmniSurface has no Usd surface output."""
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    try:
        shader.SetSourceAsset("OmniSurface.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniSurface")
    except Exception:
        shader.CreateIdAttr("OmniSurface")
    shader.CreateInput("diffuse_reflection_weight", Sdf.ValueTypeNames.Float).Set(0.18)
    shader.CreateInput("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f).Set(tuple(WATER_DIFFUSE))
    shader.CreateInput("specular_reflection_weight", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("specular_reflection_color", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
    shader.CreateInput("specular_reflection_ior", Sdf.ValueTypeNames.Float).Set(1.33)
    shader.CreateInput("specular_reflection_roughness", Sdf.ValueTypeNames.Float).Set(float(WATER_ROUGHNESS))
    shader.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular_transmission_weight", Sdf.ValueTypeNames.Float).Set(0.0)
    try:
        shader.CreateInput("enable_specular_transmission", Sdf.ValueTypeNames.Bool).Set(False)
    except Exception:
        pass
    try:
        shader.CreateInput("specular_transmission_scattering_depth", Sdf.ValueTypeNames.Float).Set(4.0)
        shader.CreateInput("specular_transmission_color", Sdf.ValueTypeNames.Color3f).Set((0.05, 0.22, 0.28))
    except Exception:
        pass
    try:
        shader.CreateInput("enable_specular_fresnel", Sdf.ValueTypeNames.Bool).Set(True)
    except Exception:
        pass
    nrm_in = None
    for name in ("geometry_normal", "normal", "bump_normal"):
        try:
            nrm_in = shader.CreateInput(name, Sdf.ValueTypeNames.Normal3f)
            break
        except Exception:
            nrm_in = None
    if tex and nrm_in is not None:
        _connect_chop_normal(stage, path, shader, nrm_in, tex)
    try:
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    except Exception:
        pass
    try:
        mat.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "out")
    except Exception:
        try:
            mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "surface")
        except Exception:
            pass
    print("water mdl OmniSurface IOR=1.33")
    return True


def make_water_material(stage, normal_tex: str | None = None, path: str = WATER_MAT) -> bool:
    """UsdPreviewSurface water. OmniSurface hid the mesh on Isaac 6 RTX (no valid surface)."""
    from pxr import Sdf, UsdShade

    global _WATER_SCROLL, _WATER_FILE
    _WATER_SCROLL = None
    _WATER_FILE = None

    png = default_normal_tex()
    png = os.path.abspath(png).replace("\\", "/") if png else None
    if png and not os.path.isfile(png):
        png = None
    tex = normal_tex
    if tex is None:
        # ROS camera Hydra product does not sample omni.ui dynamic://.
        # File PNG is visible to both the orbit viewport and /asv/camera.
        tex = png
        try:
            _upload_chop_dynamic(_chop_fft().normal_rgb_u8())
        except Exception:
            pass
        if tex:
            print("water chop file", tex)
    has_tex = bool(tex)

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    # Opaque deep water: HDR + high specular/clearcoat was reading as silver cloth.
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(WATER_DIFFUSE))
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(tuple(WATER_SPECULAR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(WATER_ROUGHNESS))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    try:
        # Specular workflow: IOR Fresnel at grazing + F0 when looking down.
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    except Exception:
        pass
    try:
        shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.25)
    except Exception:
        pass

    if has_tex:
        nrm_in = shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
        _connect_chop_normal(stage, path, shader, nrm_in, tex)
    else:
        print("water normal tex missing:", tex)

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    print("water preview UsdPreviewSurface")
    return has_tex


def bind_water_material(stage, prim_or_path) -> None:
    from pxr import UsdShade

    prim = prim_or_path if hasattr(prim_or_path, "GetPath") else stage.GetPrimAtPath(prim_or_path)
    mat = stage.GetPrimAtPath(WATER_MAT)
    if prim.IsValid() and mat.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(mat))


def _set_float2(attr, u: float, v: float) -> None:
    try:
        from pxr import Gf

        attr.Set(Gf.Vec2f(float(u), float(v)))
        return
    except Exception:
        pass
    try:
        attr.Set((float(u), float(v)))
    except Exception:
        pass


def sync_water_material(t: float, origin_xy=None) -> None:
    """Evolve chop FFT (dynamic://) and world-lock UV by camera/grid origin."""
    global _CHOP_T
    t = float(t)
    tile = max(float(DEFAULT_UV_TILE_M), 1e-3)
    if _CHOP_T is None or abs(t - _CHOP_T) > 1e-6:
        _CHOP_T = t
        fft = _chop_fft()
        fft.evolve(t)
        # Keep the shader on a file PNG. Switching to dynamic:// makes the
        # ROS camera miss the normal map and the sea turns into a mirror.
        _upload_chop_dynamic(fft.normal_rgb_u8())
    if _WATER_SCROLL is None:
        return
    if origin_xy is None:
        ox = oy = 0.0
    else:
        ox = float(origin_xy[0])
        oy = float(origin_xy[1])
    _set_float2(_WATER_SCROLL, ox / tile, oy / tile)


def _bind_fabric(path: str):
    import omni.usd
    from usdrt import Sdf, Usd, Vt

    ctx = omni.usd.get_context()
    stage_id = ctx.get_stage_id()
    rt = Usd.Stage.Attach(int(stage_id))
    prim = rt.GetPrimAtPath(path)
    if not prim:
        return None
    if not prim.HasAttribute("points"):
        return None
    if not prim.HasAttribute("Deformable"):
        tag = getattr(Sdf.ValueTypeNames, "PrimTypeTag", None) or Sdf.ValueTypeNames.Token
        prim.CreateAttribute("Deformable", tag, True)
    nrm_type = getattr(Sdf.ValueTypeNames, "Normal3fArray", None) or Sdf.ValueTypeNames.Float3Array
    if not prim.HasAttribute("normals"):
        try:
            prim.CreateAttribute("normals", nrm_type, True)
        except Exception:
            pass
    nrm_attr = prim.GetAttribute("normals") if prim.HasAttribute("normals") else None
    return prim.GetAttribute("points"), nrm_attr, Vt


class OceanMesh:
    def __init__(
        self,
        stage,
        path: str,
        waves: WaveField,
        size: float = OCEAN_SIZE_M,
        n: int = OCEAN_GRID_N,
        uv_tile_m: float = DEFAULT_UV_TILE_M,
        inner: float = 0.0,
        rest_z_bias: float = 0.0,
        min_wavelength: float = 0.0,
        use_fft: bool | None = None,
        use_wake: bool | None = None,
    ):
        from pxr import Gf, Sdf, UsdGeom, Vt

        self.path = path
        far = float(inner) > 1e-6
        if use_fft is None:
            use_fft = not far
        if use_wake is None:
            use_wake = not far
        if far and min_wavelength <= 1e-9:
            min_wavelength = float(FAR_OCEAN_MIN_LAM_M)
        if abs(float(rest_z_bias)) > 1e-12 or min_wavelength > 1e-9 or not use_fft:
            trains = list(waves.trains)
            if min_wavelength > 1e-9:
                trains = [w for w in trains if float(w.wavelength) >= float(min_wavelength)]
            waves = replace(
                waves,
                trains=trains,
                rest_z=float(waves.rest_z) + float(rest_z_bias),
                fft=None if not use_fft else waves.fft,
                _gpu=None,
            )
        self.waves = waves
        self._use_fft = bool(use_fft) and getattr(waves, "fft", None) is not None
        self._use_wake = bool(use_wake)
        self._n = int(n)
        self._size = float(size)
        self._fade_outer = 0.5 * float(size) if self._use_fft else 0.0
        self._fade_inner = max(self._fade_outer - 4.0, 0.0) if self._use_fft else 0.0
        if far:
            self.rest, counts, indices = make_ring_grid(inner, size, n)
            print(f"far ocean {float(size):.0f}m n={int(n)} hole={float(inner):.0f} min_lam={float(min_wavelength):.1f}")
        else:
            self.rest, counts, indices = make_rest_grid(size, n)
        self._gpu = None
        try:
            from .warp_gerstner import WarpGerstnerGrid, _warp

            if _warp() is not None:
                self._gpu = WarpGerstnerGrid(self.waves, self.rest, min_wavelength=min_wavelength)
                print("Warp Gerstner bound", int(self.rest.shape[0]), "verts")
                if (not far) and getattr(self.waves, "_gpu", None) is None:
                    self.waves._gpu = self._gpu
        except Exception as exc:
            print("Warp Gerstner bind skip:", exc)
        self._Gf = Gf
        self._Vt = Vt
        self._rt_points = None
        self._rt_normals = None
        self._rt_vt = None
        self._rt_tries = 0
        self._write_mode = "usd"
        self._t_op = None
        self._origin = np.zeros(2, dtype=np.float64)

        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        mesh.CreateDoubleSidedAttr().Set(True)
        mesh.CreateFaceVertexCountsAttr().Set(counts)
        mesh.CreateFaceVertexIndicesAttr().Set(indices)
        pts, nrm = waves.deform_with_normals(self.rest, 0.0)
        self._points_attr = mesh.CreatePointsAttr()
        _set_vec3f(self._points_attr, pts, Gf, Vt)
        self._normals_attr = mesh.CreateNormalsAttr()
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        _set_vec3f(self._normals_attr, nrm, Gf, Vt)
        uvs = self.rest[:, :2] / max(float(uv_tile_m), 1e-3)
        primvars = UsdGeom.PrimvarsAPI(mesh)
        st = primvars.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        _set_vec2f(st.GetAttr(), uvs, Gf, Vt)
        # PreviewSurface ignores tangent-space chop without authored tangents
        # (flat rest pose + no MikkT → mirror lake, long hull streaks).
        tangents = np.zeros((int(self.rest.shape[0]), 3), dtype=np.float32)
        tangents[:, 0] = 1.0
        for name in ("tangents", "tangent"):
            try:
                tan = primvars.CreatePrimvar(
                    name, Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex
                )
                _set_vec3f(tan.GetAttr(), tangents, Gf, Vt)
            except Exception:
                pass
        amp = float(sum(abs(w.amplitude) for w in waves.trains)) + 0.12
        mesh.CreateExtentAttr().Set(
            [Gf.Vec3f(-0.5 * size - amp, -0.5 * size - amp, -amp), Gf.Vec3f(0.5 * size + amp, 0.5 * size + amp, amp)]
        )
        prim = mesh.GetPrim()
        if not prim.HasAttribute("Deformable"):
            prim.CreateAttribute("Deformable", Sdf.ValueTypeNames.Token, True)
        try:
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            self._t_op = xf.AddTranslateOp()
            self._t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        except Exception as exc:
            print("ocean translate op skip:", exc)
        bind_water_material(stage, prim)
        self._mesh = mesh

    def _ensure_fabric(self) -> bool:
        if self._rt_points is not None:
            return True
        if self._rt_tries >= 8:
            return False
        self._rt_tries += 1
        try:
            bound = _bind_fabric(self.path)
        except Exception as exc:
            if self._rt_tries == 1:
                print("USDRT ocean bind failed:", exc)
            return False
        if bound is None:
            return False
        self._rt_points, self._rt_normals, self._rt_vt = bound
        self._write_mode = "fabric"
        print("ocean write: Fabric GPU (USDRT)")
        return True

    def _write_usd(self, pts, nrm) -> None:
        _set_vec3f(self._points_attr, pts, self._Gf, self._Vt)
        _set_vec3f(self._normals_attr, nrm, self._Gf, self._Vt)

    def sync(self, t: float, origin_xy=None, wake=None) -> None:
        if origin_xy is None:
            ox = oy = 0.0
        else:
            o = snap_origin(origin_xy)
            ox, oy = float(o[0]), float(o[1])
        self._origin[0], self._origin[1] = ox, oy
        if self._t_op is not None:
            try:
                self._t_op.Set(self._Gf.Vec3d(ox, oy, 0.0))
            except Exception:
                pass
        sync_water_material(t, (ox, oy))
        if self._use_fft and getattr(self.waves, "fft", None) is not None:
            self.waves.fft.evolve(float(t))
        gpu = self._gpu
        if gpu is None:
            gpu = getattr(self.waves, "_gpu", None)
            if gpu is not None and int(getattr(gpu, "n", -1)) != int(self.rest.shape[0]):
                gpu = None
        if gpu is not None:
            if self._use_fft:
                gpu.set_fft(self.waves)
            gpu.launch(
                t,
                origin_xy=(ox, oy),
                wake=wake if self._use_wake else None,
                fade_inner=self._fade_inner,
                fade_outer=self._fade_outer,
            )
            if self._ensure_fabric():
                try:
                    self._rt_points.Set(self._rt_vt.Vec3fArray(gpu.out))
                    if self._rt_normals is not None:
                        try:
                            self._rt_normals.Set(self._rt_vt.Vec3fArray(gpu.nrm))
                        except Exception:
                            self._write_usd(gpu.out.numpy(), gpu.nrm.numpy())
                    return
                except Exception as exc:
                    print("Fabric Set failed, USD fallback:", exc)
                    self._rt_points = None
                    self._rt_tries = 99
                    self._write_mode = "usd"
            self._write_usd(gpu.out.numpy(), gpu.nrm.numpy())
            return
        rest = self.rest
        if abs(ox) > 1e-9 or abs(oy) > 1e-9:
            rest = np.array(self.rest, copy=True)
            rest[:, 0] = rest[:, 0] + np.float32(ox)
            rest[:, 1] = rest[:, 1] + np.float32(oy)
        pts, nrm = self.waves.deform_with_normals(rest, t)
        if abs(ox) > 1e-9 or abs(oy) > 1e-9:
            pts = np.array(pts, copy=True)
            pts[:, 0] = pts[:, 0] - np.float32(ox)
            pts[:, 1] = pts[:, 1] - np.float32(oy)
        if self._use_wake and wake is not None:
            from .wake import wake_height

            xy = np.stack([self.rest[:, 0] + ox, self.rest[:, 1] + oy], axis=1)
            pts = np.array(pts, copy=True)
            pts[:, 2] = pts[:, 2] + wake_height(xy, wake).astype(np.float32)
        self._write_usd(pts, nrm)


DEFAULT_HDR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "asv_models",
    "kloofendal_48d_partly_cloudy_puresky_4k.hdr",
)
DOME_INTENSITY = 700.0
SUN_INTENSITY = 2000.0
SUN_SPECULAR = 0.95
# Real sun disc is 0.53°. Wider disc = more glints under a top-down camera.
SUN_ANGLE_DEG = 4.0
# DistantLight only. HDR sky disc stays ~48°. ~86° puts glitter under a nadir view.
SUN_GLINT_ELEV_DEG = 86.0
# Kloofendal 48d puresky peak, Z-up latlong (texture top = zenith).
_SUN = np.array([0.377476, -0.554743, 0.741466], dtype=np.float64)
DEFAULT_SUN_DIR = _SUN / float(np.linalg.norm(_SUN))

_RTX_HDR = (
    ("/rtx/shadows/enabled", True),
    ("/rtx/directLighting/enabled", True),
    ("/rtx/directLighting/sampledLighting/enabled", True),
    ("/rtx/directLighting/sampledLighting/samplesPerPixel", 2),
    ("/rtx/reflections/enabled", True),
    ("/rtx/indirectDiffuse/enabled", True),
    ("/rtx/ambientOcclusion/enabled", True),
    # Full IBL so the latlong (clouds, sun disc) is used, not a low-frequency fake sky.
    ("/rtx/domeLight/upperLowerStrategy", 0),
    ("/rtx/directLighting/domeLight/enabled", True),
    # ROS rgb8 drops alpha. If compositing punches a zero-alpha hole, the sky goes black.
    ("/rtx/post/backgroundZeroAlpha/enabled", False),
    ("/rtx/raytracing/fractionalCutoutOpacity", True),
    ("/rtx/post/tonemap/enable", True),
    # ACES: HDR dome + sun otherwise blows the hull to white.
    ("/rtx/post/tonemap/op", 2),
    ("/rtx/post/tonemap/whiteScale", 20.0),
    ("/rtx/post/tonemap/enableAutoExposure", False),
    # DLSS Quality; Performance mip-blurs the chop map when the camera is close.
    ("/rtx/post/dlss/execMode", 2),
    # Fog off. Previous session may have left it on; force-clear both RTX schemas.
    ("/rtx/post/fog/enabled", False),
    ("/rtx/fog/enabled", False),
)


def dome_zup_quat() -> np.ndarray:
    """+90° about X: USD dome texture-top (+Y) -> stage up (+Z)."""
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    return np.array([c, s, 0.0, 0.0])


def _set_rtx_usdlux_version(prim, version: int = 2505) -> None:
    """Kit 110: 2505 = OpenEXR horizon-forward. Unset defaults to 2411 (looks at HDR bottom)."""
    from pxr import Sdf

    try:
        prim.CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(int(version))
    except Exception:
        pass


def _define_sky_dome(stage, path: str = "/World/Sky"):
    """Old DomeLight + UsdLux 25.05 + rotateX 90.

    The v1 dome schema painted a black sky on Isaac 6 RTX. Quaternion
    xformOp:orient is ignored (clouds stay sideways). Use rotateX instead.
    usdluxVersion=2505 stops the 2411 'camera looks at the bottom of the HDR'
    mapping, which on a puresky is black.
    """
    from pxr import UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, path)
    prim = dome.GetPrim()
    _set_rtx_usdlux_version(prim, 2505)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    try:
        dome.OrientToStageUpAxis()
        return dome, "DomeLight usdlux=2505 OrientToStageUpAxis"
    except Exception as exc:
        print("OrientToStageUpAxis failed:", exc)
    xf.AddRotateXOp().Set(90.0)
    return dome, "DomeLight usdlux=2505 rotateX=90"


def latlong_uv_to_zup(u: float, v: float) -> np.ndarray:
    """Equirectangular UV → unit vector, Z-up (texture top = +Z)."""
    theta = float(v) * math.pi
    phi = (float(u) - 0.5) * 2.0 * math.pi
    st = math.sin(theta)
    d = np.array([st * math.sin(phi), -st * math.cos(phi), math.cos(theta)], dtype=np.float64)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def quat_wxyz_from_to(a, b) -> np.ndarray:
    """Unit quaternion (wxyz) rotating vector a onto b."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / na
    b = b / nb
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / float(np.linalg.norm(axis))
        return np.array([0.0, axis[0], axis[1], axis[2]])
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    return q / float(np.linalg.norm(q))


def _decode_rgbe_scanline(f, width: int) -> np.ndarray:
    head = f.read(4)
    if len(head) < 4:
        raise RuntimeError("truncated HDR scanline")
    scan_w = (head[2] << 8) | head[3]
    if width >= 8 and head[0] == 2 and head[1] == 2 and scan_w == width:
        chans = [bytearray(width) for _ in range(4)]
        for c in range(4):
            x = 0
            buf = chans[c]
            while x < width:
                run = f.read(1)
                if not run:
                    raise RuntimeError("truncated HDR RLE")
                n = run[0]
                if n > 128:
                    count = n - 128
                    val = f.read(1)
                    if not val:
                        raise RuntimeError("truncated HDR RLE")
                    buf[x : x + count] = bytes(val) * count
                    x += count
                else:
                    chunk = f.read(n)
                    if len(chunk) < n:
                        raise RuntimeError("truncated HDR RLE")
                    buf[x : x + n] = chunk
                    x += n
        raw = np.stack([np.frombuffer(chans[i], dtype=np.uint8) for i in range(4)], axis=1)
    else:
        rest = f.read(4 * width - 4)
        raw = np.frombuffer(head + rest, dtype=np.uint8).reshape(width, 4)
    rgb = np.zeros((width, 3), dtype=np.float32)
    e = raw[:, 3].astype(np.int32)
    valid = e > 0
    if np.any(valid):
        scale = np.ldexp(1.0, e[valid] - 136).astype(np.float32)
        rgb[valid] = raw[valid, :3].astype(np.float32) * scale[:, None]
    return rgb


def sun_dir_from_hdr(hdr_path: str) -> np.ndarray | None:
    """Brightest latlong pixel as a Z-up unit vector (texture top = zenith)."""
    if not hdr_path or not os.path.isfile(hdr_path):
        return None
    try:
        with open(hdr_path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    return None
                if line.strip() == b"":
                    break
            res = f.readline().decode("ascii", "replace").split()
            height = int(res[1])
            width = int(res[3])
            best = -1.0
            bu = bv = 0.5
            for y in range(height):
                rgb = _decode_rgbe_scanline(f, width)
                lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
                x = int(np.argmax(lum))
                v = float(lum[x])
                if v > best:
                    best = v
                    bu = (x + 0.5) / width
                    bv = (y + 0.5) / height
        if best <= 0.0:
            return None
        return latlong_uv_to_zup(bu, bv)
    except Exception as exc:
        print("HDR sun dir failed:", exc)
        return None


def enable_rtx_hdr_lighting() -> bool:
    try:
        import carb

        settings = carb.settings.get_settings()
    except Exception:
        return False
    ok = True
    aces = False
    try:
        settings.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
    except Exception:
        pass
    for key, value in _RTX_HDR:
        try:
            settings.set(key, value)
            if "tonemap/op" in key:
                aces = True
        except Exception:
            ok = False
    print(
        "lighting",
        "aces" if aces else "no-aces",
        "no-fog",
        f"dome={DOME_INTENSITY:.0f}",
        f"sun={SUN_INTENSITY:.0f}",
    )
    return ok


def _enable_light_shadows(prim, enabled: bool = True) -> None:
    from pxr import Sdf, UsdLux

    try:
        api = UsdLux.ShadowAPI.Apply(prim)
        api.CreateShadowEnableAttr(bool(enabled))
        return
    except Exception:
        pass
    try:
        prim.CreateAttribute("inputs:shadow:enable", Sdf.ValueTypeNames.Bool).Set(bool(enabled))
    except Exception:
        pass


def _set_light_common(light, intensity: float, *, specular: float = 1.0, normalize: bool = True) -> None:
    from pxr import UsdLux

    light.CreateIntensityAttr(float(intensity))
    try:
        api = UsdLux.LightAPI(light.GetPrim())
        api.CreateSpecularAttr(float(specular))
        api.CreateDiffuseAttr(1.0)
        api.CreateExposureAttr(0.0)
        api.CreateNormalizeAttr(bool(normalize))
    except Exception:
        pass


def add_dome_light(stage, hdr_path: str, intensity: float = DOME_INTENSITY) -> bool:
    if not hdr_path or not os.path.isfile(hdr_path):
        print("HDR missing:", hdr_path)
        return False
    from pxr import Sdf, UsdGeom, UsdLux

    try:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    except Exception:
        pass
    dome, orient = _define_sky_dome(stage, "/World/Sky")
    # Keep HDR range: normalize flattens a latlong dome into a dull sky.
    # Specular < 1 so the hull does not mirror the sun disc into white.
    _set_light_common(dome, intensity, specular=0.75, normalize=False)
    dome.CreateTextureFileAttr(hdr_path.replace("\\", "/"))
    try:
        dome.CreateTextureFormatAttr("latlong")
    except Exception:
        pass
    try:
        dome.CreateGuideRadiusAttr(50.0)
    except Exception:
        pass
    try:
        api = UsdLux.LightAPI(dome.GetPrim())
        if hasattr(api, "CreateVisibleInPrimaryRayAttr"):
            api.CreateVisibleInPrimaryRayAttr(True)
    except Exception:
        pass
    for name in ("inputs:visibleInPrimaryRay", "visibleInPrimaryRay"):
        try:
            dome.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(True)
        except Exception:
            pass
    _enable_light_shadows(dome.GetPrim(), False)
    print("dome light", hdr_path, orient)
    return True


def lift_sun_elevation(direction, elev_deg: float) -> np.ndarray:
    """Keep azimuth, set elevation. Used so glitter is not stuck on the horizon."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-12 else DEFAULT_SUN_DIR.copy()
    xy = d[:2]
    h = float(np.linalg.norm(xy))
    if h < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    xy = xy / h
    e = math.radians(float(elev_deg))
    out = np.array([xy[0] * math.cos(e), xy[1] * math.cos(e), math.sin(e)], dtype=np.float64)
    return out / float(np.linalg.norm(out))


def add_sun_light(stage, direction, intensity: float = SUN_INTENSITY, angle_deg: float = SUN_ANGLE_DEG):
    from pxr import Gf, UsdGeom, UsdLux

    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-12 else DEFAULT_SUN_DIR
    q = quat_wxyz_from_to((0.0, 0.0, -1.0), -d)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    _set_rtx_usdlux_version(sun.GetPrim(), 2505)
    _set_light_common(sun, intensity, specular=float(SUN_SPECULAR), normalize=True)
    try:
        sun.CreateAngleAttr(float(angle_deg))
    except Exception:
        pass
    try:
        api = UsdLux.LightAPI(sun.GetPrim())
        api.CreateEnableColorTemperatureAttr(True)
        api.CreateColorTemperatureAttr(5400.0)
    except Exception:
        pass
    xf = UsdGeom.Xformable(sun.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
    _enable_light_shadows(sun.GetPrim(), True)
    return sun


def setup_hdr_lighting(
    stage,
    hdr_path: str,
    *,
    dome_intensity: float = DOME_INTENSITY,
    sun_intensity: float = SUN_INTENSITY,
) -> bool:
    """HDR IBL sky + DistantLight sun aligned to the HDR peak, with RTX shadows."""
    enable_rtx_hdr_lighting()
    hdr_ok = add_dome_light(stage, hdr_path, intensity=dome_intensity)
    sun_dir = sun_dir_from_hdr(hdr_path) if hdr_ok else None
    if sun_dir is None:
        sun_dir = DEFAULT_SUN_DIR
        if hdr_ok:
            print("HDR sun dir fallback", sun_dir)
    hdr_elev = math.degrees(math.asin(float(np.clip(sun_dir[2], -1.0, 1.0))))
    glint_dir = lift_sun_elevation(sun_dir, SUN_GLINT_ELEV_DEG)
    add_sun_light(stage, glint_dir, intensity=sun_intensity)
    print(
        "hdr lighting",
        "ok" if hdr_ok else "no-hdr",
        "sun",
        [round(float(x), 3) for x in glint_dir],
        f"hdr_elev={hdr_elev:.1f}",
        f"glint_elev={SUN_GLINT_ELEV_DEG:.1f}",
        f"sun_angle={SUN_ANGLE_DEG:.1f}",
    )
    return hdr_ok
