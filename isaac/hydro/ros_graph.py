"""OmniGraph for ASV sensors, truth odometry and external wrench actuation."""

from __future__ import annotations

from dataclasses import dataclass, field

GRAPH = "/World/ASV_ROS"


def _wxyz_to_ijkr(q) -> list[float]:
    w, x, y, z = [float(v) for v in q]
    return [x, y, z, w]


def enable_ros_bridge(simulation_app) -> bool:
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()
        return True
    except Exception as exc:
        print("ROS2 bridge not enabled:", exc)
        return False


@dataclass
class AsvRosGraph:
    graph: str = GRAPH
    namespace: str = "asv"
    imu_prim: str = ""
    tf_prims: tuple[str, ...] = field(default_factory=tuple)
    target_odom_slugs: tuple[str, ...] = field(default_factory=tuple)
    enable_wrench: bool = False
    enable_gnss: bool = False

    @property
    def odom_attr(self) -> str:
        return f"{self.graph}/PublishOdom"

    @property
    def wrench_attr(self) -> str:
        return f"{self.graph}/SubscribeWrench"

    @property
    def gnss_attr(self) -> str:
        return f"{self.graph}/PublishGnss"

    def target_odom_attr(self, slug: str) -> str:
        return f"{self.graph}/PublishOdom_{slug}"

    def build(self) -> None:
        import omni.graph.core as og
        import omni.usd
        import usdrt.Sdf

        ns = self.namespace.strip("/")
        keys = og.Controller.Keys
        spec = {"graph_path": self.graph, "evaluator_name": "execution"}
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath(self.graph).IsValid():
            stage.RemovePrim(self.graph)

        def _edits(tick_src: str) -> dict:
            create = [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ]
            connect = [
                (tick_src, "PublishClock.inputs:execIn"),
                (tick_src, "PublishOdom.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishOdom.inputs:context"),
            ]
            values = [
                ("Context.inputs:useDomainIDEnvVar", True),
                ("Context.inputs:domain_id", 0),
                ("PublishClock.inputs:topicName", "clock"),
                ("PublishOdom.inputs:topicName", "odom"),
                ("PublishOdom.inputs:nodeNamespace", ns),
                ("PublishOdom.inputs:odomFrameId", "odom"),
                ("PublishOdom.inputs:chassisFrameId", "base_link"),
                ("PublishOdom.inputs:robotFront", [1.0, 0.0, 0.0]),
            ]
            if self.enable_wrench:
                create.append(("SubscribeWrench", "isaacsim.ros2.bridge.ROS2Subscriber"))
                connect += [
                    (tick_src, "SubscribeWrench.inputs:execIn"),
                    ("Context.outputs:context", "SubscribeWrench.inputs:context"),
                ]
                values += [
                    ("SubscribeWrench.inputs:topicName", "control_wrench"),
                    ("SubscribeWrench.inputs:nodeNamespace", ns),
                    ("SubscribeWrench.inputs:messagePackage", "geometry_msgs"),
                    ("SubscribeWrench.inputs:messageSubfolder", "msg"),
                    ("SubscribeWrench.inputs:messageName", "WrenchStamped"),
                ]
            if self.enable_gnss:
                create += [
                    ("GnssGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                    ("PublishGnss", "isaacsim.ros2.bridge.ROS2Publisher"),
                ]
                connect += [
                    (tick_src, "GnssGate.inputs:execIn"),
                    ("GnssGate.outputs:execOut", "PublishGnss.inputs:execIn"),
                    ("Context.outputs:context", "PublishGnss.inputs:context"),
                ]
                values += [
                    ("GnssGate.inputs:step", 6),
                    ("PublishGnss.inputs:topicName", "gnss/fix"),
                    ("PublishGnss.inputs:nodeNamespace", ns),
                    ("PublishGnss.inputs:messagePackage", "sensor_msgs"),
                    ("PublishGnss.inputs:messageSubfolder", "msg"),
                    ("PublishGnss.inputs:messageName", "NavSatFix"),
                ]
            for slug in self.target_odom_slugs:
                node = f"PublishOdom_{slug}"
                create.append((node, "isaacsim.ros2.bridge.ROS2PublishOdometry"))
                connect += [
                    (tick_src, f"{node}.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", f"{node}.inputs:timeStamp"),
                    ("Context.outputs:context", f"{node}.inputs:context"),
                ]
                values += [
                    (f"{node}.inputs:topicName", "odom"),
                    (f"{node}.inputs:nodeNamespace", f"{ns}/targets/{slug}"),
                    (f"{node}.inputs:odomFrameId", "odom"),
                    (f"{node}.inputs:chassisFrameId", f"{slug}_link"),
                    (f"{node}.inputs:robotFront", [1.0, 0.0, 0.0]),
                ]
            if self.imu_prim:
                create += [
                    ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                    ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ]
                connect += [
                    (tick_src, "ReadIMU.inputs:execIn"),
                    ("ReadIMU.outputs:execOut", "PublishImu.inputs:execIn"),
                    ("ReadIMU.outputs:linAcc", "PublishImu.inputs:linearAcceleration"),
                    ("ReadIMU.outputs:angVel", "PublishImu.inputs:angularVelocity"),
                    ("ReadIMU.outputs:orientation", "PublishImu.inputs:orientation"),
                    ("ReadSimTime.outputs:simulationTime", "PublishImu.inputs:timeStamp"),
                    ("Context.outputs:context", "PublishImu.inputs:context"),
                ]
                values += [
                    ("ReadIMU.inputs:imuPrim", [usdrt.Sdf.Path(self.imu_prim)]),
                    ("ReadIMU.inputs:readGravity", True),
                    ("PublishImu.inputs:topicName", "imu"),
                    ("PublishImu.inputs:nodeNamespace", ns),
                    ("PublishImu.inputs:frameId", "imu_link"),
                ]
            if self.tf_prims:
                create += [
                    ("ComputeTF", "isaacsim.core.nodes.IsaacComputeTransformTree"),
                    ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ]
                connect += [
                    (tick_src, "ComputeTF.inputs:execIn"),
                    ("ComputeTF.outputs:execOut", "PublishTF.inputs:execIn"),
                    ("ComputeTF.outputs:parentFrames", "PublishTF.inputs:parentFrames"),
                    ("ComputeTF.outputs:childFrames", "PublishTF.inputs:childFrames"),
                    ("ComputeTF.outputs:translations", "PublishTF.inputs:translations"),
                    ("ComputeTF.outputs:orientations", "PublishTF.inputs:orientations"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ("Context.outputs:context", "PublishTF.inputs:context"),
                ]
                values += [
                    ("ComputeTF.inputs:targetPrims", [usdrt.Sdf.Path(p) for p in self.tf_prims]),
                    ("PublishTF.inputs:topicName", "tf"),
                    ("PublishTF.inputs:nodeNamespace", ""),
                ]
            return {
                keys.CREATE_NODES: create,
                keys.CONNECT: connect,
                keys.SET_VALUES: values,
            }

        try:
            og.Controller.edit(spec, _edits("OnTick.outputs:tick"))
        except Exception as exc:
            print("ROS graph tick port failed, retrying execOut:", exc)
            if stage and stage.GetPrimAtPath(self.graph).IsValid():
                stage.RemovePrim(self.graph)
            og.Controller.edit(spec, _edits("OnTick.outputs:execOut"))
        og.Controller.attribute(f"{self.graph}/PublishClock.inputs:nodeNamespace").set("")
        og.Controller.attribute(f"{self.graph}/PublishClock.inputs:topicName").set("clock")
        topics = [f"/{ns}/odom"]
        if self.imu_prim:
            topics.append(f"/{ns}/imu")
        if self.enable_wrench:
            topics.append(f"/{ns}/control_wrench")
        if self.enable_gnss:
            topics.append(f"/{ns}/gnss/fix")
        for slug in self.target_odom_slugs:
            topics.append(f"/{ns}/targets/{slug}/odom")
        print(f"ROS graph {self.graph}: /clock  " + "  ".join(topics))

    @staticmethod
    def _get_dynamic(path: str):
        import omni.graph.core as og

        attr = og.Controller.attribute(path)
        return None if attr is None else attr.get()

    @staticmethod
    def _set_dynamic(path: str, value) -> bool:
        import omni.graph.core as og

        try:
            attr = og.Controller.attribute(path)
            if attr is None:
                return False
            attr.set(value)
            return True
        except Exception:
            return False

    def read_wrench(self) -> tuple[float, float, float] | None:
        """Return F, N and ROS header time; None until a complete message arrives."""
        try:
            force = self._get_dynamic(f"{self.wrench_attr}.outputs:wrench:force:x")
            moment = self._get_dynamic(f"{self.wrench_attr}.outputs:wrench:torque:z")
            sec = self._get_dynamic(f"{self.wrench_attr}.outputs:header:stamp:sec")
            nsec = self._get_dynamic(f"{self.wrench_attr}.outputs:header:stamp:nanosec")
        except Exception:
            return None
        if force is None or moment is None or sec is None or nsec is None:
            return None
        return float(force), float(moment), float(sec) + 1.0e-9 * float(nsec)

    def write_gnss(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        stamp: float,
        covariance: float = 0.0,
    ) -> bool:
        """Update the generic NavSatFix publisher fields (published by its 10 Hz gate)."""
        sec = int(max(0.0, float(stamp)))
        nsec = int(round((max(0.0, float(stamp)) - sec) * 1.0e9))
        if nsec >= 1_000_000_000:
            sec, nsec = sec + 1, 0
        node = self.gnss_attr
        ok = self._set_dynamic(f"{node}.inputs:latitude", float(latitude))
        ok = self._set_dynamic(f"{node}.inputs:longitude", float(longitude)) and ok
        ok = self._set_dynamic(f"{node}.inputs:altitude", float(altitude)) and ok
        self._set_dynamic(f"{node}.inputs:header:stamp:sec", sec)
        self._set_dynamic(f"{node}.inputs:header:stamp:nanosec", nsec)
        self._set_dynamic(f"{node}.inputs:header:frame_id", "gnss_link")
        self._set_dynamic(f"{node}.inputs:status:status", 0)
        self._set_dynamic(f"{node}.inputs:status:service", 1)
        self._set_dynamic(
            f"{node}.inputs:position_covariance",
            [float(covariance), 0.0, 0.0, 0.0, float(covariance), 0.0, 0.0, 0.0, float(covariance)],
        )
        self._set_dynamic(f"{node}.inputs:position_covariance_type", 2)
        return ok

    def write_odom(self, position, quat_wxyz, linear, angular, slug: str | None = None) -> None:
        import omni.graph.core as og

        node = self.odom_attr if slug is None else self.target_odom_attr(slug)
        pos = [float(v) for v in position]
        lin = [float(v) for v in linear]
        ang = [float(v) for v in angular]
        og.Controller.attribute(f"{node}.inputs:position").set(pos)
        og.Controller.attribute(f"{node}.inputs:orientation").set(_wxyz_to_ijkr(quat_wxyz))
        og.Controller.attribute(f"{node}.inputs:linearVelocity").set(lin)
        og.Controller.attribute(f"{node}.inputs:angularVelocity").set(ang)
