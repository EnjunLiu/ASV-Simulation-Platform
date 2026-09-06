"""Still-water floating cube for Isaac Sim 6.0.1 (experimental API)."""

from __future__ import annotations

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Short smoke run")
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

# Make the sibling `hydro` package importable when launched via python.bat
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": bool(args.headless)})

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube, DistantLight
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

from hydro.system import HydroBody, HydroSystem
from hydro.water import WaterState

CUBE_SIZE = 1.0
CUBE_MASS = 400.0
START_Z = 1.5
WATER_Z = 0.0


def _np(arr):
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.asarray(arr)


stage_utils.create_new_stage()
SimulationManager.set_physics_dt(1.0 / 60.0)

distant_light = DistantLight("/World/DistantLight")
distant_light.set_intensities(800)

water_mat = PreviewSurfaceMaterial("/Visual_materials/water")
water_mat.set_input_values("diffuseColor", [0.05, 0.35, 0.7])
water_mat.set_input_values("opacity", 0.35)

cube_mat = PreviewSurfaceMaterial("/Visual_materials/cube")
cube_mat.set_input_values("diffuseColor", [0.85, 0.35, 0.1])

# Visual water slab, no collision so the cube can sit in it.
water_shape = Cube(
    paths="/World/Water",
    positions=[0.0, 0.0, WATER_Z - 0.05],
    sizes=1.0,
    scales=[40.0, 40.0, 0.1],
)
water_shape.apply_visual_materials(water_mat)
water_geom = GeomPrim(paths=water_shape.paths, apply_collision_apis=True)
water_geom.set_enabled_collisions([False])

# Safety floor far below; buoyancy should keep the cube off it.
floor = Cube(
    paths="/World/SafetyFloor",
    positions=[0.0, 0.0, -8.0],
    sizes=1.0,
    scales=[40.0, 40.0, 0.2],
)
GeomPrim(paths=floor.paths, apply_collision_apis=True)

cube_shape = Cube(
    paths="/World/FloatCube",
    positions=[0.0, 0.0, START_Z],
    sizes=CUBE_SIZE,
    scales=[1.0, 1.0, 1.0],
)
cube_shape.apply_visual_materials(cube_mat)
cube = RigidPrim(paths=cube_shape.paths, masses=[CUBE_MASS])
GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)

water = WaterState(water_z=WATER_Z)
hydro = HydroSystem(water=water)
hydro.add(cube, HydroBody(half_extents=np.array([0.5, 0.5, 0.5]), cd=1.2, angular_damping=6.0, heave_damping_ratio=0.45, mass=CUBE_MASS))


def on_physics_step(dt: float, context: object = None) -> None:
    hydro.physics_step(dt, context)


app_utils.play()
simulation_app.update()
cube.set_masses([CUBE_MASS])
cube.set_sleep_thresholds([0.0])
SimulationManager.register_callback(on_physics_step, IsaacEvents.PRE_PHYSICS_STEP)

print("Close the window to exit." if not args.test else "test mode")
last_print = -1.0
step = 0
while True:
    if args.headless:
        SimulationManager.step()
    else:
        if not args.test and not simulation_app.is_running():
            break
        simulation_app.update()
    step += 1
    t = SimulationManager.get_simulation_time()
    if t - last_print >= 1.0 or (args.test and step % 30 == 0):
        last_print = t
        pos, _ = cube.get_world_poses()
        pos = _np(pos).reshape(-1, 3)
        z = float(pos[0, 2])
        dbg = hydro.last_debug[0] if hydro.last_debug else {}
        print(
            f"t={t:6.2f}s  z={z:7.3f}  "
            f"V={dbg.get('volume', 0):.3f}  frac={dbg.get('frac', 0):.2f}  "
            f"Fb={dbg.get('buoyancy_z', 0):.1f}N"
        )
    if args.test and step >= 90:
        break
    if args.headless and step >= 600:
        break

pos, _ = cube.get_world_poses()
final_z = float(_np(pos).reshape(-1, 3)[0, 2])
print(f"final_z={final_z:.3f}  expected_eq_z~0.10")

app_utils.stop()
simulation_app.close()

if args.test:
    # After a short drop it should be heading toward the water, not still at start height.
    if final_z > 1.4:
        raise SystemExit(f"cube did not fall into the water, z={final_z}")
