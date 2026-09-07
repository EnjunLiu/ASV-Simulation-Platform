"""Register the ASV task, then invoke Isaac Lab's stock RSL-RL launcher."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import isaaclab

import asv_sim_lab  # noqa: F401  (registers the external Gym task)

isaac_lab_root = Path(isaaclab.__file__).resolve().parents[3]
launcher = isaac_lab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not launcher.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL launcher not found: {launcher}")
sys.path.insert(0, str(launcher.parent))
runpy.run_path(str(launcher), run_name="__main__")
