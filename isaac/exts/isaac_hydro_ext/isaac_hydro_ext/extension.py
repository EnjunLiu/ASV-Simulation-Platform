"""Enable still-water hydro on Play for every dynamic rigid body in the open stage."""

from __future__ import annotations

import os
import sys

import carb
import omni.ext
import omni.timeline
import omni.usd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._hydro = None
        self._cb = None
        self._play_sub = None
        self._stop_sub = None
        try:
            import carb.eventdispatcher

            disp = carb.eventdispatcher.get_eventdispatcher()
            self._play_sub = disp.observe_event(
                event_name=omni.timeline.GLOBAL_EVENT_PLAY,
                on_event=self._on_play,
                observer_name="isaac_hydro_ext.play",
            )
            self._stop_sub = disp.observe_event(
                event_name=omni.timeline.GLOBAL_EVENT_STOP,
                on_event=self._on_stop,
                observer_name="isaac_hydro_ext.stop",
            )
        except Exception as exc:
            carb.log_warn(f"[isaac_hydro] timeline subscribe failed: {exc}")
        carb.log_info("[isaac_hydro] extension started; hydro arms on Play")

    def on_shutdown(self) -> None:
        self._disarm()
        self._play_sub = None
        self._stop_sub = None

    def _on_play(self, _event=None) -> None:
        self._arm()

    def _on_stop(self, _event=None) -> None:
        self._disarm()

    def _arm(self) -> None:
        self._disarm()
        try:
            from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

            from hydro.system import HydroSystem
            from hydro.water import WaterState
        except Exception as exc:
            carb.log_error(f"[isaac_hydro] import failed: {exc}")
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            carb.log_warn("[isaac_hydro] no stage")
            return
        self._hydro = HydroSystem.from_stage(stage, water=WaterState())
        names = [item.path for item in self._hydro.items]
        carb.log_info(f"[isaac_hydro] tracking {len(names)} bodies: {names}")

        def on_step(dt: float, context: object = None, hydro=self._hydro) -> None:
            if hydro is not None:
                hydro.physics_step(dt, context)

        try:
            SimulationManager.register_callback(on_step, IsaacEvents.PRE_PHYSICS_STEP)
            self._cb = on_step
        except Exception as exc:
            carb.log_error(f"[isaac_hydro] callback failed: {exc}")

    def _disarm(self) -> None:
        if self._cb is not None:
            try:
                from isaacsim.core.simulation_manager import SimulationManager

                if hasattr(SimulationManager, "deregister_callback"):
                    SimulationManager.deregister_callback(self._cb)
                elif hasattr(SimulationManager, "unregister_callback"):
                    SimulationManager.unregister_callback(self._cb)
            except Exception:
                pass
            self._cb = None
        self._hydro = None
