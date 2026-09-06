"""Several rigid bodies; hydro is auto-discovered from the stage."""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

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

from hydro.system import HydroSystem
from hydro.water import WaterState

WATER_Z = 0.0


def _np(arr):
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.asarray(arr)


stage_utils.create_new_stage()
SimulationManager.set_physics_dt(1.0 / 60.0)

light = DistantLight("/World/DistantLight")
light.set_intensities(800)

water_mat = PreviewSurfaceMaterial("/Visual_materials/water")
water_mat.set_input_values("diffuseColor", [0.05, 0.35, 0.7])
water_mat.set_input_values("opacity", 0.35)
water_shape = Cube(paths="/World/Water", positions=[0.0, 0.0, WATER_Z - 0.05], sizes=1.0, scales=[40.0, 40.0, 0.1])
water_shape.apply_visual_materials(water_mat)
water_geom = GeomPrim(paths=water_shape.paths, apply_collision_apis=True)
water_geom.set_enabled_collisions([False])

floor = Cube(paths="/World/SafetyFloor", positions=[0.0, 0.0, -8.0], sizes=1.0, scales=[40.0, 40.0, 0.2])
GeomPrim(paths=floor.paths, apply_collision_apis=True)

specs = [
    ("/World/LightBox", [-2.0, 0.0, 1.6], 200.0, [0.2, 0.8, 0.3]),
    ("/World/MidBox", [0.0, 0.0, 1.6], 400.0, [0.85, 0.35, 0.1]),
    ("/World/HeavyBox", [2.0, 0.0, 1.6], 800.0, [0.6, 0.2, 0.2]),
]
for path, pos, mass, color in specs:
    mat = PreviewSurfaceMaterial("/Visual_materials/" + path.split("/")[-1])
    mat.set_input_values("diffuseColor", color)
    shape = Cube(paths=path, positions=pos, sizes=1.0, scales=[1.0, 1.0, 1.0])
    shape.apply_visual_materials(mat)
    RigidPrim(paths=shape.paths, masses=[mass])
    GeomPrim(paths=shape.paths, apply_collision_apis=True)

app_utils.play()
simulation_app.update()

stage = stage_utils.get_current_stage()
hydro = HydroSystem.from_stage(stage, water=WaterState(water_z=WATER_Z))
print("hydro bodies:", [item.path for item in hydro.items])


def on_physics_step(dt: float, context: object = None) -> None:
    hydro.physics_step(dt, context)


SimulationManager.register_callback(on_physics_step, IsaacEvents.PRE_PHYSICS_STEP)

print("Close the window to exit.")
step = 0
last = 0.0
while args.headless or simulation_app.is_running():
    if args.headless:
        SimulationManager.step()
        step += 1
        if step > 600:
            break
    else:
        simulation_app.update()
    t = SimulationManager.get_simulation_time()
    if t - last >= 1.0:
        last = t
        line = []
        for item in hydro.items:
            pos, _ = item.rigid_prim.get_world_poses()
            z = float(_np(pos).reshape(-1, 3)[0, 2])
            line.append(f"{item.path.split('/')[-1]} z={z:.2f}")
        print(f"t={t:5.1f}  " + "  ".join(line))

app_utils.stop()
simulation_app.close()
