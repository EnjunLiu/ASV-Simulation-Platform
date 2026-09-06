"""Full-dynamics ASV scene for Isaac Sim.

The ego vessel is always a six-degree-of-freedom rigid body driven by forces at
the two physical propeller locations.  This simulator contains no perception,
policy, guidance-law or controller implementation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "asv_models"

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--smoke", action="store_true", help="Run a three-second physics smoke and exit")
parser.add_argument("--seconds", type=float, default=0.0, help="Exit after this many simulation seconds; zero keeps running")
parser.add_argument("--mass", type=float, default=18.0)
parser.add_argument("--thrust", type=float, default=3.0, help="Manual forward thrust per propeller, N")
parser.add_argument("--diff", type=float, default=0.0, help="Manual right-minus-left thrust, N")
parser.add_argument("--actuation", choices=("manual", "wrench"), default="manual")
parser.add_argument("--prop-half-spacing", type=float, default=0.11)
parser.add_argument("--prop-max-thrust", type=float, default=3.0)
parser.add_argument("--wrench-timeout", type=float, default=0.25)
parser.add_argument("--spin-delay", type=float, default=2.0)
parser.add_argument("--amp", type=float, default=1.0, help="Sea-state scale; zero is still water")
parser.add_argument("--hydro-cd", type=float, default=1.1)
parser.add_argument("--hydro-cm", type=float, default=1.0)
parser.add_argument("--buoyancy-coefficient", type=float, default=1.15)
parser.add_argument("--heave-damping-ratio", type=float, default=0.5)
parser.add_argument("--horizontal-damping-ratio", type=float, default=0.25)
parser.add_argument("--surge-damping-ratio", type=float, default=None)
parser.add_argument("--sway-damping-ratio", type=float, default=None)
parser.add_argument("--angular-damping", type=float, default=14.0)
parser.add_argument("--layout", default="none", help="none, T1, or L7B")
parser.add_argument("--motion", default="S0", help="S0, S1, or S2 target motion")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hull-color", choices=("yellow", "red", "blue", "white", "gray"), default="yellow")
parser.add_argument("--usd", default=None)
parser.add_argument("--hdr", default=str(ASSETS / "kloofendal_48d_partly_cloudy_puresky_4k.hdr"))
parser.add_argument("--ros", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ns", default="asv")
parser.add_argument("--no-camera", action="store_true")
parser.add_argument("--no-imu", action="store_true")
parser.add_argument("--gnss-latitude", type=float, default=31.2304)
parser.add_argument("--gnss-longitude", type=float, default=121.4737)
parser.add_argument("--gnss-altitude", type=float, default=0.0)
parser.add_argument("--gnss-noise", type=float, default=0.0)
parser.add_argument("--enhanced-determinism", action=argparse.BooleanOptionalAction, default=False)
args, _ = parser.parse_known_args()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydro.scene_layout import LAYOUT_IDS, TARGET_ODOM_SLUG, layout_poses, target_prim_path
from hydro.target_motion import motion_dir, parse_motion, planar_cmd_armed
from hydro.usd_asv import usd_for_hull_color

layout = str(args.layout).strip().upper()
if layout in ("", "NONE", "OFF"):
    layout = "none"
elif layout == "T1":
    layout = "L7B"
elif layout not in LAYOUT_IDS:
    raise SystemExit(f"unknown layout {args.layout!r}; expected none, T1 or {LAYOUT_IDS}")
args.motion = parse_motion(args.motion)
args.usd = str(args.usd or usd_for_hull_color(args.hull_color))

if args.ros:
    from hydro.ros_setup import prepare_ros_env

    prepare_ros_env()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": bool(args.headless)})

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

from hydro.discover import origin_in_frame
from hydro.mesh import HullMesh
from hydro.ocean import (
    DOME_INTENSITY,
    FAR_OCEAN_GRID_N,
    FAR_OCEAN_INNER_M,
    FAR_OCEAN_SIZE_M,
    FAR_OCEAN_Z_BIAS,
    OCEAN_GRID_N,
    OCEAN_SIZE_M,
    OceanMesh,
    camera_xy_world,
    make_water_material,
    setup_hdr_lighting,
    snap_origin,
)
from hydro.ros_graph import AsvRosGraph, enable_ros_bridge
from hydro.ros_sensors import (
    CAMERA_PRIM,
    IMU_PRIM,
    attach_forward_camera,
    attach_imu,
    build_camera_graph,
    enable_sensor_extensions,
)
from hydro.system import HydroBody, HydroSystem, read_rigid_state
from hydro.thrust import Thruster, ThrusterBank
from hydro.usd_asv import find_hull_mesh_path
from hydro.wake import ShipWake
from hydro.water import WaterState
from hydro.waves import mild_sea
from hydro.wrench_control import WrenchWatchdog, mix_force_moment, world_xy_to_navsat

WATER_Z = 0.0
ASV_PATH = "/World/ASV"
ROOT_PATH = "/World/ASV/ASV_Root"
LEFT_PROP = f"{ROOT_PATH}/Left_Propeller"
RIGHT_PROP = f"{ROOT_PATH}/Right_Propeller"
FALLBACK_PROP = {
    "left": np.array([-0.09813, 0.11000, -0.20176]),
    "right": np.array([-0.09813, -0.11000, -0.20176]),
}


def _np(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _mass_properties(hull: HullMesh, mass: float):
    com = np.asarray(hull.cob_local, dtype=np.float64).copy()
    com[2] -= 0.03
    hx, hy, hz = [2.0 * float(v) for v in hull.half_extents]
    m = float(mass)
    return com, m / 12.0 * (hy * hy + hz * hz), m / 12.0 * (hx * hx + hz * hz), m / 12.0 * (hx * hx + hy * hy)


def _mount_asv(stage, prim_path: str, usd: str, mass: float):
    from pxr import Gf, UsdPhysics

    if not os.path.isfile(usd):
        raise FileNotFoundError(usd)
    stage_utils.add_reference_to_stage(usd_path=usd, path=prim_path)
    root_path = f"{prim_path}/ASV_Root"
    root_prim = stage.GetPrimAtPath(root_path)
    hull_path = find_hull_mesh_path(stage, root_path)
    hull_prim = stage.GetPrimAtPath(hull_path)
    if not root_prim.IsValid() or not hull_prim.IsValid():
        raise RuntimeError(f"invalid ASV USD hierarchy below {prim_path}")
    hull_geom = GeomPrim(paths=hull_path, apply_collision_apis=True)
    hull_geom.set_collision_approximations(["convexHull"])
    hull_geom.set_enabled_collisions([False])
    hull = HullMesh.from_prims(hull_prim, root_prim)
    com, ixx, iyy, izz = _mass_properties(hull, mass)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    api = UsdPhysics.MassAPI.Apply(root_prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(float(ixx), float(iyy), float(izz)))
    return {
        "root_path": root_path,
        "hull_path": hull_path,
        "root_prim": root_prim,
        "rigid": RigidPrim(paths=root_path, masses=[mass]),
        "hull": hull,
        "com": com,
    }


def _hydro_body(hull: HullMesh):
    return HydroBody(
        mesh=hull,
        mass=float(args.mass),
        cd=float(args.hydro_cd),
        cm=float(args.hydro_cm),
        cb=float(args.buoyancy_coefficient),
        angular_damping=float(args.angular_damping),
        heave_damping_ratio=float(args.heave_damping_ratio),
        horizontal_damping_ratio=float(args.horizontal_damping_ratio),
        surge_damping_ratio=args.surge_damping_ratio,
        sway_damping_ratio=args.sway_damping_ratio,
    )


def _settle(spec, xy, z, label):
    rigid, hull = spec["rigid"], spec["hull"]
    com, ixx, iyy, izz = _mass_properties(hull, args.mass)
    rigid.set_masses([args.mass])
    rigid.set_sleep_thresholds([0.0])
    rigid.set_coms(positions=[com.tolist()])
    rigid.set_inertias([[ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz]])
    rigid.set_world_poses(positions=[[float(xy[0]), float(xy[1]), float(z)]])
    rigid.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])
    geom = GeomPrim(paths=spec["hull_path"], apply_collision_apis=True)
    geom.set_collision_approximations(["convexHull"])
    geom.set_enabled_collisions([True])
    print(f"ASV {label}: mass={args.mass:.1f} kg CoM={com} hull={spec['hull_path']}")


def _replace_yaw(quat, yaw):
    w, x, y, z = [float(v) for v in quat]
    current = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * (float(yaw) - current)
    c, s = math.cos(half), math.sin(half)
    return [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]


def _apply_planar(rigid, state, vx, vy, yaw):
    quat = _replace_yaw(state.orientation, yaw)
    rigid.set_world_poses(positions=[[float(v) for v in state.position]], orientations=[quat])
    rigid.set_velocities(
        linear_velocities=[[float(vx), float(vy), float(state.linear[2])]],
        angular_velocities=[[float(state.angular[0]), float(state.angular[1]), 0.0]],
    )
    return quat


ros_graph = None
if args.ros and enable_ros_bridge(simulation_app):
    enable_sensor_extensions()
    simulation_app.update()
else:
    args.ros = False

stage_utils.create_new_stage()
SimulationManager.set_physics_dt(1.0 / 60.0)
floor = Cube(paths="/World/SafetyFloor", positions=[0.0, 0.0, -6.0], sizes=1.0, scales=[50.0, 50.0, 0.2])
GeomPrim(paths=floor.paths, apply_collision_apis=True)
stage = stage_utils.get_current_stage()
if args.enhanced_determinism:
    from pxr import Sdf

    for scene in (p for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene"):
        scene.CreateAttribute("physxScene:enableEnhancedDeterminism", Sdf.ValueTypeNames.Bool).Set(True)

ego = _mount_asv(stage, ASV_PATH, args.usd, args.mass)
asv, hull = ego["rigid"], ego["hull"]
make_water_material(stage)
setup_hdr_lighting(stage, args.hdr, dome_intensity=DOME_INTENSITY)

camera_prim = attach_forward_camera(stage, path=CAMERA_PRIM) if args.ros and not args.no_camera else None
imu_prim = attach_imu(IMU_PRIM) if args.ros and not args.no_imu else None
simulation_app.update()

waves = mild_sea(args.amp)
water = WaterState(water_z=WATER_Z, waves=waves if abs(args.amp) > 1.0e-6 else None)
ocean = OceanMesh(stage, "/World/Ocean", waves, size=OCEAN_SIZE_M, n=OCEAN_GRID_N)
ocean_far = OceanMesh(stage, "/World/OceanFar", waves, size=FAR_OCEAN_SIZE_M, n=FAR_OCEAN_GRID_N, inner=FAR_OCEAN_INNER_M, rest_z_bias=FAR_OCEAN_Z_BIAS)
for path in ("/World/Ocean", "/World/OceanFar"):
    try:
        GeomPrim(paths=path, apply_collision_apis=True).set_enabled_collisions([False])
    except Exception:
        pass

hydro = HydroSystem(water=water)
hydro.add(asv, _hydro_body(hull), path=ROOT_PATH)
spawn_z = WATER_Z - float(hull.aabb_min[2]) + 0.12

targets = []
if layout != "none":
    for pose in layout_poses(layout):
        spec = _mount_asv(stage, target_prim_path(pose.prim_name), usd_for_hull_color(pose.color), args.mass)
        spec["pose"] = pose
        targets.append(spec)
        hydro.add(spec["rigid"], _hydro_body(spec["hull"]), path=spec["root_path"])
    print(f"target layout={args.layout} motion={args.motion} seed={args.seed} dir={motion_dir(args.seed):+d}")

left = origin_in_frame(stage.GetPrimAtPath(LEFT_PROP), ego["root_prim"])
right = origin_in_frame(stage.GetPrimAtPath(RIGHT_PROP), ego["root_prim"])
left = FALLBACK_PROP["left"].copy() if left is None or abs(float(left[1])) < 1.0e-4 else left
right = FALLBACK_PROP["right"].copy() if right is None or abs(float(right[1])) < 1.0e-4 else right
thrusters = ThrusterBank([
    Thruster("left", left, np.array([1.0, 0.0, 0.0]), max_force=args.prop_max_thrust),
    Thruster("right", right, np.array([1.0, 0.0, 0.0]), max_force=args.prop_max_thrust),
])
wrench = WrenchWatchdog(float(args.wrench_timeout))
wrench_debug = {"valid": False, "left": 0.0, "right": 0.0, "saturated": 0, "samples": 0}
gnss_bucket = -1
gnss_noise = np.zeros(3)
gnss_rng = np.random.default_rng(args.seed)

if args.ros:
    try:
        tf_prims = [ROOT_PATH] + ([camera_prim] if camera_prim else []) + ([imu_prim] if imu_prim else [])
        ros_graph = AsvRosGraph(
            namespace=args.ns,
            imu_prim=imu_prim or "",
            tf_prims=tuple(tf_prims),
            target_odom_slugs=tuple(TARGET_ODOM_SLUG[s["pose"].entity_id] for s in targets),
            enable_wrench=args.actuation == "wrench",
            enable_gnss=True,
        )
        ros_graph.build()
        if camera_prim:
            build_camera_graph(camera_prim, namespace=args.ns)
    except Exception as exc:
        print(f"ROS graph unavailable: {exc}")
        ros_graph = None


def _manual_commands(t):
    if t < args.spin_delay:
        return {"left": 0.0, "right": 0.0}
    return {"left": args.thrust - 0.5 * args.diff, "right": args.thrust + 0.5 * args.diff}


def on_physics_step(dt, _context=None):
    global gnss_bucket, gnss_noise
    t = SimulationManager.get_simulation_time()
    water.time = t
    if args.actuation == "wrench":
        message = ros_graph.read_wrench() if ros_graph is not None else None
        if message is not None:
            wrench.update(*message)
        force, moment, valid = wrench.command(t)
        mixed = mix_force_moment(force, moment, half_spacing=args.prop_half_spacing, max_thrust=args.prop_max_thrust)
        commands = {"left": mixed.left, "right": mixed.right} if t >= args.spin_delay else {"left": 0.0, "right": 0.0}
        wrench_debug.update(valid=valid, left=commands["left"], right=commands["right"])
        wrench_debug["samples"] += 1
        wrench_debug["saturated"] += int(valid and mixed.saturated)
    else:
        commands = _manual_commands(t)
    thrusters.set_commands(commands)

    states = [read_rigid_state(item.rigid_prim) for item in hydro.items]
    hydro.physics_step(dt, _context, states=states)
    ego_state = states[0] if states else None
    thrusters.physics_step(asv, state=ego_state)

    for spec, state in zip(targets, states[1:]):
        if state is None:
            continue
        vx, vy, yaw = planar_cmd_armed(args.motion, spec["pose"].entity_id, args.seed, t, args.spin_delay)
        quat = _apply_planar(spec["rigid"], state, vx, vy, yaw)
        if ros_graph is not None:
            ros_graph.write_odom(state.position, quat, [vx, vy, float(state.linear[2])], [float(state.angular[0]), float(state.angular[1]), 0.0], slug=TARGET_ODOM_SLUG[spec["pose"].entity_id])

    if ros_graph is not None and ego_state is not None:
        ros_graph.write_odom(ego_state.position, ego_state.orientation, ego_state.linear, ego_state.angular)
        bucket = int(max(0.0, t) * 10.0)
        if bucket != gnss_bucket:
            gnss_bucket = bucket
            gnss_noise = gnss_rng.normal(0.0, max(0.0, args.gnss_noise), size=3)
        noisy = np.asarray(ego_state.position) + gnss_noise
        lat, lon, alt = world_xy_to_navsat(noisy[0], noisy[1], noisy[2], origin_latitude=args.gnss_latitude, origin_longitude=args.gnss_longitude, origin_altitude=args.gnss_altitude)
        ros_graph.write_gnss(lat, lon, alt, t, covariance=args.gnss_noise ** 2)


app_utils.play()
simulation_app.update()
_settle(ego, (0.0, 0.0), spawn_z, "ego")
for spec in targets:
    _settle(spec, (spec["pose"].x, spec["pose"].y), spawn_z, spec["pose"].entity_id)
SimulationManager.register_callback(on_physics_step, IsaacEvents.PRE_PHYSICS_STEP)


def _wake():
    state = read_rigid_state(asv)
    if state is None:
        return None
    w, x, y, z = [float(v) for v in state.orientation]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return ShipWake(float(state.position[0]), float(state.position[1]), math.cos(yaw), math.sin(yaw), float(np.hypot(state.linear[0], state.linear[1])))


def _sync_ocean():
    camera = camera_xy_world()
    if camera is None:
        position, _ = asv.get_world_poses()
        camera = _np(position).reshape(-1, 3)[0, :2]
    origin = snap_origin(camera)
    ocean.sync(SimulationManager.get_simulation_time(), origin_xy=origin, wake=_wake())
    ocean_far.sync(SimulationManager.get_simulation_time(), origin_xy=origin, wake=None)


print("ASV_SIM_READY full_dynamics=true controller_embedded=false")
last_print = -1.0
start_wall = time.monotonic()
start_sim = SimulationManager.get_simulation_time()
limit = 3.0 if args.smoke else max(0.0, args.seconds)
while args.headless or simulation_app.is_running():
    simulation_app.update()
    t = SimulationManager.get_simulation_time()
    if not args.headless:
        _sync_ocean()
    if args.actuation == "wrench":
        ahead = (t - start_sim) - (time.monotonic() - start_wall)
        if ahead > 0.0:
            time.sleep(min(ahead, 0.02))
    if t - last_print >= 1.0:
        last_print = t
        state = read_rigid_state(asv)
        debug = hydro.last_debug[0] if hydro.last_debug else {}
        if state is not None:
            commands = wrench_debug if args.actuation == "wrench" else _manual_commands(t)
            print(f"t={t:6.2f}s pos=({state.position[0]:+.2f},{state.position[1]:+.2f},{state.position[2]:+.2f}) L={commands['left']:+.2f} R={commands['right']:+.2f} Fb={debug.get('buoyancy_z', 0.0):.1f}N")
    if limit > 0.0 and t - start_sim >= limit:
        break

app_utils.stop()
simulation_app.close()
print("ASV_SIM_DONE")
