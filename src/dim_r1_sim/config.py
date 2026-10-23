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

"""Everything about an R1Sim run: the scene, the robot's pose, the physics, and each sensor."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SceneConfig(BaseModel):
    # office_555's folder (scene_office_555.xml, meshes/, textures/), e.g. a dimos checkout's
    # data/mujoco_office_555; None downloads dimOS's archive of it once into physics.cache_dir.
    office_dir: Path | None = None
    # x, y, yaw of the robot: office_555's origin is where dimOS's MuJoCo sim spawns its robot.
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)


class RobotConfig(BaseModel):
    # The R1 Pro's URDF; None is the one dimOS drives the real robot with (Galaxea's, pinned in
    # dimos.robot.galaxea.r1pro.config), fetched once into dimOS's robot asset cache.
    urdf: Path | None = None
    # Joint angles (radians, by URDF name) it stands at; every other joint is at 0: torso upright,
    # arms hanging at its sides, as the vendor's zero pose is.
    joint_positions: dict[str, float] = Field(default_factory=dict)


class PhysicsConfig(BaseModel):
    timestep: float = 0.005  # seconds per MuJoCo step
    control_dt: float = 0.02  # seconds between reading cmd_vel and setting the base's actuators
    # The base's limits, vx, vy m/s and wz rad/s, and their accelerations: dimOS's
    # R1PRO_PLANAR_BASE (dimos.robot.galaxea.r1pro.config).
    max_velocity: tuple[float, float, float] = (1.0, 1.0, 2.0)
    max_acceleration: tuple[float, float, float] = (2.0, 2.0, 4.0)
    command_timeout: float = 0.5  # seconds a cmd_vel holds before the base stops
    realtime: bool = True  # pace to the wall clock; False runs as fast as it can
    cache_dir: Path | None = None  # compiled worlds, the office and URDF files; None is ~/.cache


class LidarConfig(BaseModel):
    enabled: bool = True
    frame: str = "lidar_chassis_left_link"  # the chassis Mid-360, as the URDF mounts it
    rate_hz: float = 10.0
    point_rate_hz: int = 200_000
    downsample: int = 1
    min_range: float = 0.16
    max_range: float = 40.0
    noise: bool = True
    dropout: bool = True
    self_occlusion: bool = True  # the robot's own body (torso, arms, wheels) blocks rays
    motion_bins_hz: float = 200.0  # how often a scan's origin follows the moving base
    seed: int = 0


class CameraConfig(BaseModel):
    enabled: bool = True
    rate_hz: float = 10.0
    # The head's stereo pair, as the URDF mounts their optical frames (z forward, x right, y down).
    left_frame: str = "camera_head_left_link"
    right_frame: str = "camera_head_right_link"
    right: bool = True  # draw the right eye too (on right_color_image); False is the left only
    # The ISX031's 1920x1536 mode at a third the size; fovy from its factory fy, ~1012 px of 1536.
    width: int = 640
    height: int = 512
    fovy_deg: float = 74.4


class UrdfConfig(BaseModel):
    # The robot and the office as one URDF on robot_description, each file on file_request ->
    # file_info, as dimOS's UrdfManager shares one (for dim-controller); every link goes on tf too.
    enabled: bool = True
    period_s: float = Field(default=5.0, gt=0.0)  # robot_description is re-sent; nothing is latched
    tile_faces: int = Field(default=100_000, ge=1)  # triangles per STL file (50 bytes each)
