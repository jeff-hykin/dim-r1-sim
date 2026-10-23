# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""R1Sim: a simulated Galaxea R1 Pro standing in dimOS's office, with its chassis Mid-360 and head
cameras, driven by cmd_vel."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from dim_go2_sim.urdf import MeshFile, SharedUrdf
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger
from dimos_lcm.std_msgs import String
import mujoco
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation

from dim_r1_sim.config import (
    CameraConfig,
    LidarConfig,
    PhysicsConfig,
    RobotConfig,
    SceneConfig,
    UrdfConfig,
)
from dim_r1_sim.lidar import BodyRaycaster, LidarScan, chassis_lidar, office_raycaster
from dim_r1_sim.office import office_dir
from dim_r1_sim.robot import ROOT, load_urdf, r1pro_urdf, swerve, wheel_modules
from dim_r1_sim.urdf import share_urdf
from dim_r1_sim.world import R1World, load_world

logger = setup_logger()

# robot_description goes out at most this often, however many viewers send an empty file_request.
URDF_MIN_INTERVAL_S = 0.2


class R1SimConfig(ModuleConfig):
    scene: SceneConfig = Field(default_factory=SceneConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    urdf: UrdfConfig = Field(default_factory=UrdfConfig)
    office: bool = True  # False stands it on bare ground
    odom_hz: float = 50.0
    frame_prefix: str = ""  # namespaces every frame published, e.g. "r1/"


class R1Sim(Module):
    """cmd_vel in; the R1 Pro's chassis lidar, head cameras, odometry and tf out.

    Odometry is the true base pose. The upper body stands at its configured pose; the swerve
    wheels' steer and roll on tf follow the base's velocity.
    """

    config: R1SimConfig
    dedicated_worker = True

    cmd_vel: In[Twist]
    lidar: Out[PointCloud2]
    color_image: Out[Image]  # the left head camera
    camera_info: Out[CameraInfo]
    right_color_image: Out[Image]
    right_camera_info: Out[CameraInfo]
    odom: Out[PoseStamped]
    tf: Out[TFMessage]
    # The robot and office as a URDF, its meshes named package://dim_r1_sim/<path>?hash=<sha256>;
    # any of those names on file_request gets the file on file_info, unless it has &have=<sha256>;
    # an empty file_request gets robot_description now, for a viewer that just connected.
    robot_description: Out[String]
    file_request: In[String]
    file_info: Out[MeshFile]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()  # the command
        self._physics = threading.Lock()  # the world's state, while a tick changes it
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._command = np.zeros(3)
        self._command_time = -math.inf
        self._clock_origin = 0.0
        self._shared_urdf: SharedUrdf | None = None
        self._file_requests: list[str] = []
        self._file_wanted = threading.Condition()
        self._urdf_wanted = threading.Event()
        self._wheel_angles: dict[str, float] = {}

    @rpc
    def start(self) -> None:
        super().start()
        cfg = self.config
        self.robot = load_urdf(cfg.robot.urdf or r1pro_urdf())
        self.poses = self.robot.link_poses(cfg.robot.joint_positions)
        for frame in (cfg.lidar.frame, cfg.camera.left_frame, cfg.camera.right_frame):
            if frame not in self.poses:
                raise ValueError(f"{frame} is not a link of {self.robot.name}")
        self.office = (
            office_dir(cfg.scene.office_dir, cfg.physics.cache_dir) if cfg.office else None
        )
        model = load_world(self.office, self.robot, self.poses, cfg.physics, cfg.camera)
        self.world = R1World(model, cfg.physics)
        self.world.reset(*cfg.scene.start)
        blocked = [
            (model.geom(c.geom1).name, model.geom(c.geom2).name)
            for c in self.world.data.contact[: self.world.data.ncon]
        ]
        if blocked:
            raise RuntimeError(f"the robot starts inside office geometry: {blocked}")
        self._modules = wheel_modules(self.robot)
        self._lidar = None
        if cfg.lidar.enabled:
            raycaster: Any = office_raycaster(model, self.world.data)
            if cfg.lidar.self_occlusion:  # not the lidar's own housing: it doesn't see itself
                raycaster = BodyRaycaster(
                    raycaster, self.robot, self.poses, self.world.base_pose, (cfg.lidar.frame,)
                )
            self._lidar_mount = self.poses[cfg.lidar.frame]
            self._lidar = chassis_lidar(cfg.lidar, raycaster, self._lidar_pose)
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        self._stop.clear()
        self._threads = [threading.Thread(target=self._run, name="r1-sim", daemon=True)]
        if cfg.urdf.enabled:
            self.register_disposable(Disposable(self.file_request.subscribe(self._on_file_request)))
            self._threads += [
                threading.Thread(target=self._share_urdf, daemon=True),
                threading.Thread(target=self._answer_file_requests, daemon=True),
            ]
        if cfg.camera.enabled:
            self._threads.append(threading.Thread(target=self._run_cameras, daemon=True))
        for thread in self._threads:
            thread.start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
        self._urdf_wanted.set()
        with self._file_wanted:
            self._file_wanted.notify_all()
        for thread in self._threads:
            thread.join(timeout=10)
        super().stop()

    @rpc
    def status(self) -> dict[str, float]:
        """Sim time against wall time since the physics started."""
        return {"sim_time": self.world.time, "wall_time": time.time() - self._clock_origin}

    def _frame(self, name: str) -> str:
        return self.config.frame_prefix + name

    def _on_cmd_vel(self, twist: Twist) -> None:
        with self._lock:
            self._command = np.array([twist.linear.x, twist.linear.y, twist.angular.z])
            self._command_time = time.monotonic()

    def _lidar_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        position, rotation = self.world.base_pose()
        mount = self._lidar_mount
        return position + rotation @ mount[:3, 3], rotation @ mount[:3, :3]

    def _run(self) -> None:
        cfg = self.config
        world = self.world
        self._clock_origin = time.time()
        wall_start = time.monotonic()
        next_odom = 0.0
        scans: list[LidarScan] = []

        def on_substep() -> None:
            if self._lidar is not None and (scan := self._lidar.advance(world.time)) is not None:
                scans.append(scan)

        while not self._stop.is_set():
            with self._lock:
                fresh = time.monotonic() - self._command_time <= cfg.physics.command_timeout
                command = self._command.copy() if fresh else np.zeros(3)
            with self._physics:  # the cameras copy the pose between ticks
                world.tick(command, on_substep)
            stamp = self._clock_origin + world.time
            if world.time >= next_odom:
                next_odom += 1.0 / cfg.odom_hz
                self._publish_pose(stamp)
            for scan in scans:
                self._publish_scan(scan)
            scans.clear()
            if cfg.physics.realtime:
                ahead = world.time - (time.monotonic() - wall_start)
                if ahead > 0:
                    self._stop.wait(ahead)
                elif ahead < -1.0:  # fell behind (e.g. a slow start): don't race to catch up
                    wall_start += -ahead

    def _publish_pose(self, stamp: float) -> None:
        position, rotation = self.world.base_pose()
        x, y, z, w = Rotation.from_matrix(rotation).as_quat()
        world, base = self._frame("world"), self._frame(ROOT)
        self.odom.publish(
            PoseStamped(
                ts=stamp,
                frame_id=world,
                position=Vector3(*position),
                orientation=Quaternion(x, y, z, w),
            )
        )
        transforms = [
            Transform(
                translation=Vector3(*position),
                rotation=Quaternion(x, y, z, w),
                frame_id=world,
                child_frame_id=base,
                ts=stamp,
            )
        ]
        # Every URDF joint: the upper body at its pose, the wheels steered and rolled to the base's
        # velocity.
        wheels = swerve(self._modules, self.world.body_velocity(), self._wheel_angles)
        positions = dict(self.config.robot.joint_positions)
        for name, value in wheels.items():
            if name.startswith("wheel_motor_joint"):
                value = self._wheel_angles.get(name, 0.0) + value / self.config.odom_hz
                value = math.remainder(value, 2 * math.pi)
            self._wheel_angles[name] = value
            positions[name] = value
        for joint in self.robot.joints:
            edge = joint.at(positions.get(joint.name, 0.0))
            qx, qy, qz, qw = Rotation.from_matrix(edge[:3, :3]).as_quat()
            transforms.append(
                Transform(
                    translation=Vector3(*edge[:3, 3]),
                    rotation=Quaternion(qx, qy, qz, qw),
                    frame_id=self._frame(joint.parent),
                    child_frame_id=self._frame(joint.child),
                    ts=stamp,
                )
            )
        self.tf.publish(TFMessage(*transforms))

    def _publish_scan(self, scan: LidarScan) -> None:
        self.lidar.publish(
            PointCloud2.from_numpy(
                scan.points,
                frame_id=self._frame(self.config.lidar.frame),
                timestamp=self._clock_origin + scan.start,
                offset_times=np.rint(scan.offsets * 1e9).astype(np.uint32),
                lines=scan.lines,
            )
        )

    def _run_cameras(self) -> None:
        from dim_r1_sim.camera import HeadCameras, focal_length

        cfg = self.config.camera
        model = self.world.model
        data = mujoco.MjData(model)  # the cameras' own copy, posed from the physics between ticks
        cameras = HeadCameras(model, cfg)
        focal = focal_length(cfg)
        eyes = [
            ("left", self._frame(cfg.left_frame), self.color_image, self.camera_info),
        ]
        if cfg.right:
            eyes.append(
                (
                    "right",
                    self._frame(cfg.right_frame),
                    self.right_color_image,
                    self.right_camera_info,
                )
            )
        period = 1.0 / cfg.rate_hz
        next_frame = time.monotonic()
        try:
            while not self._stop.wait(max(0.0, next_frame - time.monotonic())):
                next_frame = max(next_frame + period, time.monotonic())
                with self._physics:
                    data.qpos[:] = self.world.data.qpos
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                stamp = time.time()
                for eye, frame, image_out, info_out in eyes:
                    rgb = cameras.capture(data, eye)
                    info = CameraInfo.from_intrinsics(
                        focal, focal, cfg.width / 2, cfg.height / 2, cfg.width, cfg.height, frame
                    )
                    info.ts = stamp
                    info_out.publish(info)
                    image_out.publish(
                        Image(data=rgb, format=ImageFormat.RGB, frame_id=frame, ts=stamp)
                    )
        finally:
            cameras.close()

    def _share_urdf(self) -> None:
        """Builds the office's files (once: a few seconds), then re-sends the URDF."""
        cfg = self.config
        started = time.monotonic()
        office_key = ""
        if self.office is not None:
            xml = self.office / "scene_office_555.xml"
            office_key = f"{xml}:{xml.stat().st_mtime_ns}"
        shared = share_urdf(
            self.robot,
            self.world.model if self.office is not None else None,
            self.world.data if self.office is not None else None,
            office_key,
            cfg.urdf,
            cfg.frame_prefix,
            cfg.physics.cache_dir,
        )
        self._shared_urdf = shared
        logger.info(
            f"robot_description: {len(shared.files)} files, "
            f"{sum(p.stat().st_size for p, _ in shared.files.values()) / 1e6:.1f} MB, "
            f"ready in {time.monotonic() - started:.1f} s"
        )
        while not self._stop.is_set():
            self._urdf_wanted.clear()
            self.robot_description.publish(String(shared.xml))
            sent = time.monotonic()
            self._urdf_wanted.wait(cfg.urdf.period_s)  # an empty file_request sends it sooner
            self._stop.wait(sent + URDF_MIN_INTERVAL_S - time.monotonic())  # however many ask

    def _on_file_request(self, request: String) -> None:
        if not request.data:
            self._urdf_wanted.set()
            return
        with self._file_wanted:
            self._file_requests.append(request.data)
            self._file_wanted.notify()

    def _answer_file_requests(self) -> None:
        """One at a time, off the transport's thread: a file can be MBs."""
        while not self._stop.is_set():
            with self._file_wanted:
                while not self._file_requests and not self._stop.is_set():
                    self._file_wanted.wait()
                requests, self._file_requests = self._file_requests, []
            for request in dict.fromkeys(requests):
                shared = self._shared_urdf
                try:
                    file = shared.answer(request) if shared is not None else None
                except KeyError:
                    logger.warning(f"file_request for {request!r}: not a file in robot_description")
                    continue
                if file is not None:
                    self.file_info.publish(file)
