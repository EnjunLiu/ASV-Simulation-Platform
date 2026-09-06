"""Ship-mounted camera and IMU for the ASV ROS prototype."""

from __future__ import annotations

import math
import struct
import zlib

CAMERA_GRAPH = "/World/ASV_ROS_Camera"
CAMERA_PRIM = "/World/ASV/ASV_Root/camera_link"
IMU_PRIM = "/World/ASV/ASV_Root/imu_link"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV_DEG = 90.0
# 16:9 so fx == fy at 1280x720. Renderer assumes square pixels.
CAMERA_H_APERTURE = 16.0
CAMERA_V_APERTURE = 9.0
CAMERA_FOCAL = CAMERA_H_APERTURE / 2.0
CAMERA_MOUNT = (0.42, 0.0, 0.2)
CAMERA_PITCH_DOWN_DEG = 5.0
CAMERA_HZ = 10
PHYSICS_HZ = 60
# Ship-camera Hydra product. Viewport ACES whiteScale=20, AE off.
# Do not enable auto-exposure here: the Kloofendal sun disc sits in frame
# and AE keys off it, crushing the blue sky to black (looks like night).
# Do not author USD Camera shutter or aperture either — that crushed ROS LdrColor.
CAMERA_RP_WHITE_SCALE = 20.0


def camera_fx_fy() -> tuple[float, float]:
    fx = CAMERA_WIDTH * CAMERA_FOCAL / CAMERA_H_APERTURE
    fy = CAMERA_HEIGHT * CAMERA_FOCAL / CAMERA_V_APERTURE
    return fx, fy


def camera_usd_axes(pitch_down_deg: float = CAMERA_PITCH_DOWN_DEG):
    """USD camera basis: rows are local X/Y/Z in parent/world.

    USD cameras look along -Z with +Y up. We want -Z = body +X, pitched down.
    USD Gf matrices are row-vector (p' = p * M), so axes go in rows not columns.
    """
    pitch = math.radians(float(pitch_down_deg))
    look = (math.cos(pitch), 0.0, -math.sin(pitch))
    z_axis = (-look[0], -look[1], -look[2])
    world_up = (0.0, 0.0, 1.0)
    x_axis = (
        world_up[1] * z_axis[2] - world_up[2] * z_axis[1],
        world_up[2] * z_axis[0] - world_up[0] * z_axis[2],
        world_up[0] * z_axis[1] - world_up[1] * z_axis[0],
    )
    x_len = math.sqrt(x_axis[0] ** 2 + x_axis[1] ** 2 + x_axis[2] ** 2)
    if x_len < 1e-8:
        x_axis = (0.0, -1.0, 0.0)
    else:
        x_axis = (x_axis[0] / x_len, x_axis[1] / x_len, x_axis[2] / x_len)
    y_axis = (
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    )
    y_len = math.sqrt(y_axis[0] ** 2 + y_axis[1] ** 2 + y_axis[2] ** 2)
    y_axis = (y_axis[0] / y_len, y_axis[1] / y_len, y_axis[2] / y_len)
    z_len = math.sqrt(z_axis[0] ** 2 + z_axis[1] ** 2 + z_axis[2] ** 2)
    z_axis = (z_axis[0] / z_len, z_axis[1] / z_len, z_axis[2] / z_len)
    return x_axis, y_axis, z_axis


def rgb_to_png_bytes(rgb, compress: int = 9) -> bytes:
    """Write an HxWx3 uint8 RGB array as a PNG (stdlib only)."""
    import numpy as np

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB, got {arr.shape}")
    arr = np.ascontiguousarray(arr[:, :, :3], dtype="uint8")
    height, width, _ = arr.shape
    level = max(1, min(9, int(compress)))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, level)),
            chunk(b"IEND", b""),
        )
    )


def save_rgb_png(path: str, rgb, compress: int = 9) -> str:
    data = rgb_to_png_bytes(rgb, compress=compress)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def ros_image_to_rgb(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB."""
    import numpy as np

    height = int(msg.height)
    width = int(msg.width)
    encoding = str(getattr(msg, "encoding", "rgb8") or "rgb8").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        img = buf.reshape(height, width, 3)
        if encoding == "bgr8":
            img = img[:, :, ::-1]
        return img
    if encoding in ("rgba8", "bgra8"):
        img = buf.reshape(height, width, 4)
        rgb = img[:, :, :3]
        if encoding == "bgra8":
            rgb = rgb[:, :, ::-1]
        return rgb
    raise ValueError(f"unsupported image encoding {encoding!r}")


def grab_render_product_rgb(graph: str = CAMERA_GRAPH):
    """RGB from the same render product ROS2CameraHelper publishes."""
    import numpy as np
    import omni.graph.core as og
    import omni.replicator.core as rep

    try:
        render_product_path = og.Controller.attribute(
            f"{graph}/createRenderProduct.outputs:renderProductPath"
        ).get()
    except Exception:
        return None
    if not render_product_path:
        return None
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    paths = render_product_path if isinstance(render_product_path, (list, tuple)) else [render_product_path]
    try:
        annot.attach(paths)
    except Exception:
        try:
            annot.attach(render_product_path)
        except Exception as exc:
            print("rgb annotator attach failed:", exc)
            return None
    data = annot.get_data()
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    return arr[:, :, :3]


class PublishedImageSnap:
    """Subscribe once; poll with spin_once while the sim loop runs."""

    def __init__(self, topic: str = "/asv/camera/image_raw"):
        self.topic = topic
        self.msg = None
        self._node = None
        try:
            import rclpy
            from sensor_msgs.msg import Image
        except Exception as exc:
            print("rclpy Image subscribe skipped:", exc)
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._rclpy = rclpy
            self._node = rclpy.create_node("isaac_hydro_image_snap")
            self._node.create_subscription(Image, topic, self._cb, 10)
        except Exception as exc:
            print("rclpy Image node failed:", exc)
            self._node = None

    def _cb(self, msg) -> None:
        self.msg = msg

    def poll(self, timeout_sec: float = 0.0):
        if self._node is None:
            return None
        try:
            self._rclpy.spin_once(self._node, timeout_sec=float(timeout_sec))
        except Exception as exc:
            print("rclpy spin_once failed:", exc)
        return self.msg

    def close(self) -> None:
        if self._node is None:
            return
        try:
            self._node.destroy_node()
        except Exception:
            pass
        self._node = None


def grab_published_image(topic: str = "/asv/camera/image_raw", timeout_sec: float = 0.0):
    """One sensor_msgs/Image from the ROS 2 topic, or None."""
    snap = PublishedImageSnap(topic)
    try:
        return snap.poll(timeout_sec=timeout_sec)
    finally:
        snap.close()


def attach_forward_camera(
    stage,
    *,
    path: str = CAMERA_PRIM,
    mount=CAMERA_MOUNT,
    pitch_down_deg: float = CAMERA_PITCH_DOWN_DEG,
) -> str:
    """USD Camera under the hull. Looks along +X, pitched down, -Z is the optical axis."""
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, path)
    x_axis, y_axis, z_axis = camera_usd_axes(pitch_down_deg)
    rot = Gf.Matrix3d()
    rot.SetRow(0, Gf.Vec3d(*x_axis))
    rot.SetRow(1, Gf.Vec3d(*y_axis))
    rot.SetRow(2, Gf.Vec3d(*z_axis))
    mat = Gf.Matrix4d(rot, Gf.Vec3d(float(mount[0]), float(mount[1]), float(mount[2])))
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(mat)
    cam.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
    cam.GetHorizontalApertureAttr().Set(float(CAMERA_H_APERTURE))
    cam.GetVerticalApertureAttr().Set(float(CAMERA_V_APERTURE))
    cam.GetFocalLengthAttr().Set(float(CAMERA_FOCAL))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 10000.0))
    print(
        f"camera {CAMERA_WIDTH}x{CAMERA_HEIGHT} fov={CAMERA_HFOV_DEG} "
        f"mount={tuple(mount)} pitch_down={pitch_down_deg} {path}"
    )
    return path


def attach_imu(path: str = IMU_PRIM) -> str | None:
    try:
        from isaacsim.sensors.experimental.physics import IMU
    except Exception as exc:
        print("IMU authoring missing:", exc)
        return None
    IMU.create(path, translations=[[0.0, 0.0, 0.0]], orientations=[[1.0, 0.0, 0.0, 0.0]])
    print(f"IMU {path}")
    return path


def enable_sensor_extensions() -> None:
    from isaacsim.core.utils.extensions import enable_extension

    for name in (
        "isaacsim.sensors.physics",
        "isaacsim.sensors.experimental.physics",
        "omni.syntheticdata",
    ):
        try:
            enable_extension(name)
        except Exception as exc:
            print(f"extension {name} skipped:", exc)


def build_camera_graph(
    camera_prim: str,
    *,
    namespace: str = "asv",
    graph: str = CAMERA_GRAPH,
) -> None:
    """Ondemand RGB + CameraInfo graph. Copied from Isaac camera_periodic.py."""
    import omni.graph.core as og
    import omni.usd
    import usdrt.Sdf

    ns = namespace.strip("/")
    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath(graph).IsValid():
        stage.RemovePrim(graph)
    keys = og.Controller.Keys
    spec = {
        "graph_path": graph,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    }
    ros_camera_graph, _, _, _ = og.Controller.edit(
        spec,
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("createRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
                ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("createRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim)]),
                ("createRenderProduct.inputs:width", CAMERA_WIDTH),
                ("createRenderProduct.inputs:height", CAMERA_HEIGHT),
                ("cameraHelperRgb.inputs:frameId", "camera_link"),
                ("cameraHelperRgb.inputs:topicName", "camera/image_raw"),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("cameraHelperRgb.inputs:nodeNamespace", ns),
                ("cameraHelperInfo.inputs:frameId", "camera_link"),
                ("cameraHelperInfo.inputs:topicName", "camera/camera_info"),
                ("cameraHelperInfo.inputs:nodeNamespace", ns),
            ],
        },
    )
    og.Controller.evaluate_sync(ros_camera_graph)
    step = max(1, int(round(PHYSICS_HZ / CAMERA_HZ)))
    if _set_camera_gates(graph, step):
        print(f"camera gate step={step} ({CAMERA_HZ} Hz)")
    else:
        print("camera gate not set, publishing every frame")
    print(f"ROS camera {graph}: /{ns}/camera/image_raw  /{ns}/camera/camera_info")


def _pump_app(n: int = 1) -> None:
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(max(int(n), 1)):
            app.update()
    except Exception:
        pass


def _set_gate_step(template: str, render_product_path: str, step: int) -> bool:
    import omni.graph.core as og
    import omni.syntheticdata

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(template, render_product_path)
    try:
        og.Controller.attribute(gate_path + ".inputs:step").set(int(step))
        return True
    except Exception:
        return False


def _set_camera_gates(graph: str, step: int) -> bool:
    import omni.graph.core as og
    import omni.syntheticdata
    import omni.syntheticdata._syntheticdata as sd

    for _ in range(8):
        _pump_app()
        try:
            render_product_path = og.Controller.attribute(
                f"{graph}/createRenderProduct.outputs:renderProductPath"
            ).get()
        except Exception:
            render_product_path = None
        if not render_product_path:
            continue
        rv_rgb = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        rgb_ok = _set_gate_step(rv_rgb + "IsaacSimulationGate", render_product_path, step)
        info_ok = _set_gate_step("PostProcessDispatchIsaacSimulationGate", render_product_path, step)
        if rgb_ok and info_ok:
            _apply_render_product_tonemap(render_product_path)
            return True
    return False


def _apply_render_product_tonemap(render_product_path: str) -> bool:
    """Same ACES as the viewport, plus dome IBL on this Hydra view (AE off)."""
    path = str(render_product_path or "").strip()
    if not path:
        return False
    ok = False
    try:
        import omni.usd
        from pxr import Sdf

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            specs = (
                ("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int, 2505),
                # Hydra product already authors this as TfToken; Int 0 black-skies the rgb8 dump.
                ("omni:rtx:lights:dome:samplingStrategy", Sdf.ValueTypeNames.Token, "auto"),
                ("omni:rtx:directLighting:domeLight:enabled", Sdf.ValueTypeNames.Bool, True),
                ("omni:rtx:post:compositing:enabled", Sdf.ValueTypeNames.Bool, False),
                ("omni:rtx:post:compositing:blackBackground", Sdf.ValueTypeNames.Bool, False),
                ("omni:rtx:post:backgroundZeroAlpha:enabled", Sdf.ValueTypeNames.Bool, False),
                ("omni:rtx:post:tonemap:enable", Sdf.ValueTypeNames.Bool, True),
                ("omni:rtx:post:tonemap:op", Sdf.ValueTypeNames.Int, 2),
                ("omni:rtx:post:tonemap:whiteScale", Sdf.ValueTypeNames.Float, float(CAMERA_RP_WHITE_SCALE)),
                ("omni:rtx:post:tonemap:enableAutoExposure", Sdf.ValueTypeNames.Bool, False),
            )
            for name, typ, val in specs:
                try:
                    attr = prim.GetAttribute(name)
                    if not (attr and attr.IsValid()):
                        attr = prim.CreateAttribute(name, typ)
                    attr.Set(val)
                except Exception as exc:
                    print(f"camera rp attr skip {name}:", exc)
            ok = True
    except Exception as exc:
        print("camera rp usd tonemap skip:", exc)
    try:
        import carb

        settings = carb.settings.get_settings()
        name = path.rstrip("/").split("/")[-1]
        prefixes = ("", path, f"/hydra/renderProducts/{name}", f"/exts/omni.kit.hydra_texture/{name}")
        for prefix in prefixes:
            try:
                pfx = f"{prefix}/rtx" if prefix else "/rtx"
                settings.set(f"{pfx}/domeLight/upperLowerStrategy", 0)
                settings.set(f"{pfx}/directLighting/domeLight/enabled", True)
                settings.set(f"{pfx}/post/backgroundZeroAlpha/enabled", False)
                settings.set(f"{pfx}/post/backgroundZeroAlpha/blackBackgroundInComposite", False)
                settings.set(f"{pfx}/post/tonemap/enable", True)
                settings.set(f"{pfx}/post/tonemap/op", 2)
                settings.set(f"{pfx}/post/tonemap/whiteScale", float(CAMERA_RP_WHITE_SCALE))
                settings.set(f"{pfx}/post/tonemap/enableAutoExposure", False)
                ok = True
            except Exception:
                pass
    except Exception as exc:
        print("camera rp carb tonemap skip:", exc)
    if ok:
        print(f"camera ldr tonemap whiteScale={CAMERA_RP_WHITE_SCALE} no-auto-exposure dome-ibl {path}")
    else:
        print("camera ldr tonemap not applied", path)
    return ok


def reapply_camera_tonemap(graph: str = CAMERA_GRAPH) -> bool:
    """Reapply camera HDR settings after late Hydra initialization.

    Isaac can recreate/overwrite render-product settings shortly after graph
    construction, leaving a stable but underexposed LdrColor stream.  This is
    safe to call when a live-frame quality check detects that state.
    """
    try:
        import omni.graph.core as og

        render_product_path = og.Controller.attribute(
            f"{graph}/createRenderProduct.outputs:renderProductPath"
        ).get()
    except Exception:
        return False
    return _apply_render_product_tonemap(render_product_path)
