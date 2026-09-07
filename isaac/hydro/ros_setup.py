"""Set Isaac Sim 6 ROS 2 env before SimulationApp() so the humble bridge can load."""

from __future__ import annotations

import os
import sys


def _humble_root() -> str | None:
    tails = os.path.join("exts", "isaacsim.ros2.core", "humble")
    roots = []
    for key in ("ISAAC_PATH", "CARB_APP_PATH"):
        val = os.environ.get(key)
        if val:
            roots.append(val)
    cur = os.path.dirname(os.path.abspath(sys.executable))
    for _ in range(8):
        roots.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    seen = set()
    for root in roots:
        path = os.path.join(root, tails)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path):
            return path
    return None


def _default_fastdds_xml() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "fastdds.xml",
    )


def prepare_ros_env(
    *,
    domain_id: str = "0",
    fastdds_xml: str | None = None,
) -> None:
    os.environ["ROS_DISTRO"] = "humble"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    os.environ.setdefault("ROS_DOMAIN_ID", str(domain_id))
    os.environ["ROS_LOCALHOST_ONLY"] = "0"
    os.environ["ROS_DISCOVERY_SERVER"] = ""
    xml = fastdds_xml or _default_fastdds_xml()
    if os.path.isfile(xml):
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = xml
    humble = _humble_root()
    if humble:
        lib = os.path.join(humble, "lib")
        ros_python = os.path.join(humble, "rclpy")
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
        if ros_python not in sys.path:
            sys.path.insert(0, ros_python)
        pythonpath = os.environ.get("PYTHONPATH", "")
        if ros_python not in pythonpath.split(os.pathsep):
            os.environ["PYTHONPATH"] = ros_python + (os.pathsep + pythonpath if pythonpath else "")
        print("ROS humble lib on PATH:", lib)
    else:
        print("ROS humble lib not found; bridge may fail. Use isaac-sim.humble.bat env.")
