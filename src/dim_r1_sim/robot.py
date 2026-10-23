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

"""The R1 Pro's URDF as a kinematic tree: its links' meshes and its joints, posed at any angles.

The URDF is Galaxea's, as dimOS drives the real robot with it, so every frame (base_link,
lidar_chassis_left_link, camera_head_left_link, ...) is named as on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

ROOT = "base_link"
_STL_TRIANGLE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
MOVING = ("revolute", "continuous", "prismatic")

Transform = NDArray[np.float64]  # 4x4, child in parent


@dataclass(frozen=True)
class Visual:
    mesh: Path
    origin: Transform  # the mesh in its link's frame
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class Joint:
    name: str
    type: str  # fixed, revolute, continuous or prismatic
    parent: str
    child: str
    origin: Transform  # the child's frame in the parent's, at 0
    axis: NDArray[np.float64]
    limits: tuple[float, float] | None  # lower, upper; None for fixed and continuous joints

    def at(self, position: float) -> Transform:
        """The child's frame in the parent's with the joint at *position*."""
        if self.type not in MOVING or position == 0.0:
            return self.origin
        motion = np.eye(4)
        if self.type == "prismatic":
            motion[:3, 3] = self.axis * position
        else:
            motion[:3, :3] = Rotation.from_rotvec(self.axis * position).as_matrix()
        return self.origin @ motion


@dataclass(frozen=True)
class Robot:
    name: str
    visuals: dict[str, list[Visual]]  # link -> its meshes
    masses: dict[str, float]
    joints: list[Joint]  # parents before children

    @property
    def links(self) -> list[str]:
        return [ROOT, *(j.child for j in self.joints)]

    def joint(self, name: str) -> Joint:
        return next(j for j in self.joints if j.name == name)

    def link_poses(self, positions: dict[str, float] | None = None) -> dict[str, Transform]:
        """Each link's frame in base_link's, the joints at *positions* (radians, metres; 0 else)."""
        positions = positions or {}
        poses = {ROOT: np.eye(4)}
        for j in self.joints:
            poses[j.child] = poses[j.parent] @ j.at(positions.get(j.name, 0.0))
        return poses

    def vertices(self, link: str) -> NDArray[np.float32]:
        """Every vertex of *link*'s meshes, in its frame."""
        corners = [
            transformed(read_stl(v.mesh).reshape(-1, 3), v.origin) for v in self.visuals[link]
        ]
        return np.concatenate(corners) if corners else np.zeros((0, 3), np.float32)


def transform(xyz: str | None, rpy: str | None) -> Transform:
    result = np.eye(4)
    if rpy:
        result[:3, :3] = Rotation.from_euler("xyz", [float(a) for a in rpy.split()]).as_matrix()
    if xyz:
        result[:3, 3] = [float(a) for a in xyz.split()]
    return result


def transformed(points: NDArray[np.floating], pose: Transform) -> NDArray[np.float32]:
    return (points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def load_urdf(path: Path) -> Robot:
    """The robot a URDF describes; package://<any>/<file> meshes resolve beside the urdf folder,
    as Galaxea's package lays them out (urdf/ and meshes/ side by side)."""
    root = ET.parse(path).getroot()
    package = path.parent.parent
    visuals: dict[str, list[Visual]] = {}
    masses: dict[str, float] = {}
    for link in root.iter("link"):
        name = link.attrib["name"]
        visuals[name] = []
        if (mass := link.find("inertial/mass")) is not None:
            masses[name] = float(mass.attrib["value"])
        for visual in link.iter("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib["filename"]
            if filename.startswith("package://"):
                file = package / filename.removeprefix("package://").split("/", 1)[1]
            else:
                file = path.parent / filename
            origin = visual.find("origin")
            color = visual.find("material/color")
            rgba = (
                tuple(float(c) for c in color.attrib["rgba"].split()) if color is not None else ()
            )
            visuals[name].append(
                Visual(
                    file,
                    transform(
                        *(origin.get(k) for k in ("xyz", "rpy")) if origin is not None else ()
                    ),
                    rgba if len(rgba) == 4 else (0.7, 0.7, 0.7, 1.0),  # type: ignore[arg-type]
                )
            )
    by_parent: dict[str, list[Joint]] = {}
    for element in root.iter("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        kind = element.attrib["type"]
        joint = Joint(
            name=element.attrib["name"],
            type=kind,
            parent=element.find("parent").attrib["link"],  # type: ignore[union-attr]
            child=element.find("child").attrib["link"],  # type: ignore[union-attr]
            origin=transform(origin.get("xyz"), origin.get("rpy"))
            if origin is not None
            else np.eye(4),
            axis=np.array([float(a) for a in axis.attrib["xyz"].split()])
            if axis is not None
            else np.array([1.0, 0.0, 0.0]),
            limits=(float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            if limit is not None and kind in ("revolute", "prismatic")
            else None,
        )
        by_parent.setdefault(joint.parent, []).append(joint)
    ordered: list[Joint] = []
    pending = [ROOT]
    while pending:  # breadth first from base_link: parents before children
        for joint in by_parent.get(pending.pop(0), []):
            ordered.append(joint)
            pending.append(joint.child)
    return Robot(root.attrib.get("name", "r1pro"), visuals, masses, ordered)


def r1pro_urdf() -> Path:
    """Galaxea's R1 Pro URDF as dimOS pins it, cloned into dimOS's robot asset cache once."""
    from dimos.robot.galaxea.r1pro.config import R1PRO_MODEL_PATH

    return Path(str(R1PRO_MODEL_PATH))


@cache
def read_stl(path: Path) -> NDArray[np.float32]:
    """A binary STL's triangles, (n, 3, 3)."""
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    if 84 + count * _STL_TRIANGLE.itemsize != len(data):
        raise ValueError(f"{path}: not a binary STL")
    triangles: NDArray[np.float32] = np.frombuffer(data, _STL_TRIANGLE, count, 84)["vertices"]
    return triangles


def wheel_modules(robot: Robot) -> list[tuple[Joint, Joint, float]]:
    """Each swerve module's steer joint, its wheel joint, and the wheel's radius (from its mesh)."""
    modules = []
    for steer in robot.joints:
        if steer.name.startswith("steer_motor_joint"):
            wheel = next(j for j in robot.joints if j.parent == steer.child)
            points = robot.vertices(wheel.child)
            across = np.linalg.norm(points[:, [0, 2]] - points[:, [0, 2]].mean(0), axis=1)
            modules.append((steer, wheel, float(across.max())))
    return modules


def swerve(
    modules: list[tuple[Joint, Joint, float]],
    velocity: NDArray[np.float64],
    steer_angles: dict[str, float],
) -> dict[str, float]:
    """Each module's steer angle and wheel rate (rad/s) for a body velocity (vx, vy, wz), each
    steer kept within its limits by driving the wheel backwards; a still module keeps its angle."""
    vx, vy, wz = velocity
    result = {}
    for steer, wheel, radius in modules:
        px, py = steer.origin[:2, 3]
        mx, my = vx - wz * py, vy + wz * px
        speed = math.hypot(mx, my)
        angle = steer_angles.get(steer.name, 0.0)
        if speed > 1e-3:
            angle = math.atan2(my, mx)
            lower, upper = steer.limits or (-math.pi, math.pi)
            if not lower <= angle <= upper:
                angle -= math.copysign(math.pi, angle)
                speed = -speed
            elif abs(angle) > math.pi / 2:  # the shorter turn, rolling backwards
                angle -= math.copysign(math.pi, angle)
                speed = -speed
        result[steer.name] = angle
        result[wheel.name] = speed / radius * float(np.sign(wheel.axis[1]) or 1.0)
    return result
