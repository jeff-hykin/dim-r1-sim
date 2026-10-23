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

"""The office with the R1 Pro in it, compiled once and cached, its base driven by a velocity.

The base is a holonomic chassis on three swerve wheels; here it slides on the floor on planar
joints (x, y, yaw), each with a velocity actuator whose force is capped at the base's
acceleration limit times its mass (or inertia), so it can't tip and stops at what it runs into.
The upper body stands rigid at its pose: each link's meshes are drawn and cast at it, and two
boxes, the chassis's and the upper body's, collide with the office.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dim_r1_sim.config import CameraConfig, PhysicsConfig
from dim_r1_sim.office import cache_root, office_spec
from dim_r1_sim.robot import ROOT, Robot, Transform, transformed

PREFIX = "r1/"
BASE_JOINTS = ("base_x", "base_y", "base_yaw")
VISUAL_GROUP = 2  # drawn and cast, never collided with
HIDDEN_GROUP = 5  # neither drawn nor cast: the collision boxes
# Bump when the composed model changes, so cached worlds are rebuilt.
MODEL_VERSION = 1
# A MuJoCo camera looks down its -z with y up; an optical frame looks down +z with y down.
OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])
EYES = ("left", "right")


def robot_spec(
    robot: Robot,
    poses: dict[str, Transform],
    physics: PhysicsConfig,
    camera: CameraConfig,
) -> mujoco.MjSpec:
    """The R1 alone, its links' meshes welded to one base body at *poses*, on planar joints."""
    spec = mujoco.MjSpec()
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    mass = sum(robot.masses.values())
    base = spec.worldbody.add_body(name=ROOT)
    # Its mass and yaw inertia about the chassis centre: a 0.6 m box of the robot's mass is near.
    extent = np.ptp(robot.vertices(ROOT)[:, :2], axis=0)
    inertia = mass * (extent @ extent) / 12
    base.mass = mass
    base.inertia = [inertia, inertia, inertia]
    for axis, name in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), BASE_JOINTS, strict=True):
        kind = mujoco.mjtJoint.mjJNT_HINGE if name == "base_yaw" else mujoco.mjtJoint.mjJNT_SLIDE
        base.add_joint(name=name, type=kind, axis=axis)
    limits = zip(physics.max_velocity, physics.max_acceleration, (mass, mass, inertia), strict=True)
    for name, (speed, accel, weight) in zip(BASE_JOINTS, limits, strict=True):
        # Gain to stop within a control tick; force capped at the acceleration limit.
        spec.add_actuator(
            name=name,
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=[-speed, speed],
            forcerange=[-accel * weight, accel * weight],
            forcelimited=True,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            gainprm=[weight / physics.control_dt] + [0] * 9,
            biasprm=[0, 0, -weight / physics.control_dt] + [0] * 7,
        )
    meshes = 0
    for link, visuals in robot.visuals.items():
        for visual in visuals:
            name = f"mesh_{meshes}"
            meshes += 1
            spec.add_mesh(name=name, file=str(visual.mesh))
            pose = poses[link] @ visual.origin
            w, x, y, z = _quat(pose[:3, :3])
            base.add_geom(
                name=f"{link}_{name}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                pos=pose[:3, 3],
                quat=[w, x, y, z],
                rgba=visual.rgba,
                group=VISUAL_GROUP,
                contype=0,
                conaffinity=0,
                mass=0,
            )
    chassis, upper = collision_boxes(robot, poses)
    for name, (low, high) in (("chassis_box", chassis), ("upper_box", upper)):
        base.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(low + high) / 2,
            size=(high - low) / 2,
            rgba=[0, 0, 0, 0],
            group=HIDDEN_GROUP,
            mass=0,
        )
    for eye, frame in zip(EYES, (camera.left_frame, camera.right_frame), strict=True):
        pose = poses[frame]
        w, x, y, z = _quat(pose[:3, :3] @ OPTICAL_TO_MUJOCO)
        base.add_camera(
            name=f"head_{eye}", pos=pose[:3, 3], quat=[w, x, y, z], fovy=camera.fovy_deg
        )
    return spec


def collision_boxes(
    robot: Robot, poses: dict[str, Transform]
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
    """The chassis's bounds, lifted off the floor (its wheels roll; the box must not drag), and
    the upper body's above it: (low, high) corners in base_link."""
    chassis = robot.vertices(ROOT)
    wheels = {j.child for j in robot.joints if j.name.startswith(("steer_motor", "wheel_motor"))}
    upper = np.concatenate(
        [
            transformed(robot.vertices(link), poses[link])
            for link in robot.links
            if link != ROOT and link not in wheels and robot.visuals[link]
        ]
    )
    low, high = chassis.min(0).astype(np.float64), chassis.max(0).astype(np.float64)
    upper = upper[upper[:, 2] > high[2]]
    return (low, high), (
        np.array([*upper[:, :2].min(0), high[2]]),
        upper.max(0).astype(np.float64),
    )


def _quat(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return np.array([w, x, y, z])


def compose(
    office: Path | None,
    robot: Robot,
    poses: dict[str, Transform],
    physics: PhysicsConfig,
    camera: CameraConfig,
) -> mujoco.MjModel:
    """office_555 (or bare ground) with the R1 under PREFIX."""
    if office is not None:
        world = office_spec(office)
    else:
        world = mujoco.MjSpec()
        world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    world.option.timestep = physics.timestep
    world.attach(
        robot_spec(robot, poses, physics, camera),
        prefix=PREFIX,
        frame=world.worldbody.add_frame(),
    )
    model: mujoco.MjModel = world.compile()
    # Clip planes in metres, as dimOS's sim sets them: MuJoCo scales them by the scene's extent.
    model.vis.map.znear = 0.01 / model.stat.extent
    model.vis.map.zfar = 100 / model.stat.extent
    return model


def load_world(
    office: Path | None,
    robot: Robot,
    poses: dict[str, Transform],
    physics: PhysicsConfig,
    camera: CameraConfig,
) -> mujoco.MjModel:
    """compose(), cached by everything it is built from: the R1's meshes take seconds."""
    parts = [
        (office / "scene_office_555.xml").read_bytes() if office else b"ground",
        str(office).encode(),
        repr(sorted((k, v.round(6).tolist()) for k, v in poses.items())).encode(),
        repr(sorted(str(v.mesh) for vs in robot.visuals.values() for v in vs)).encode(),
        physics.model_dump_json(
            include={"timestep", "control_dt", "max_velocity", "max_acceleration"}
        ).encode(),
        camera.model_dump_json(include={"left_frame", "right_frame", "fovy_deg"}).encode(),
        f"{mujoco.__version__} {version('dim-r1-sim')} {MODEL_VERSION}".encode(),
    ]
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()[:16]
    path = cache_root(physics.cache_dir) / f"{'office_555' if office else 'ground'}-{key}.mjb"
    if path.is_file():
        return mujoco.MjModel.from_binary_path(str(path))
    model = compose(office, robot, poses, physics, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.tmp")
    mujoco.mj_saveModel(model, str(partial), None)
    partial.rename(path)
    return model


class R1World:
    """The compiled world and the R1's base in it, driven toward a body-frame (vx, vy, wz)."""

    def __init__(self, model: mujoco.MjModel, physics: PhysicsConfig) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.physics = physics
        joints = [model.joint(PREFIX + name).id for name in BASE_JOINTS]
        self._qpos = model.jnt_qposadr[joints]
        self._dofs = model.jnt_dofadr[joints]
        self._actuators = [model.actuator(PREFIX + name).id for name in BASE_JOINTS]
        self.base = model.body(PREFIX + ROOT).id
        self.substeps = max(1, round(physics.control_dt / model.opt.timestep))
        self.time = 0.0

    def reset(self, x: float, y: float, yaw: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos] = [x, y, yaw]
        mujoco.mj_forward(self.model, self.data)
        self.time = 0.0

    def tick(
        self, command: NDArray[np.float64], on_substep: Callable[[], None] | None = None
    ) -> None:
        """Advance one control tick toward a body-frame (vx, vy, wz) command."""
        yaw = float(self.data.qpos[self._qpos[2]])
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy, wz = command
        self.data.ctrl[self._actuators] = [c * vx - s * vy, s * vx + c * vy, wz]
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            self.time += self.model.opt.timestep
            if on_substep is not None:
                on_substep()

    def base_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Position and rotation matrix of base_link in the world."""
        return (
            self.data.xpos[self.base].copy(),
            self.data.xmat[self.base].reshape(3, 3).copy(),
        )

    def body_velocity(self) -> NDArray[np.float64]:
        """(vx, vy) in base_link and the yaw rate."""
        vx, vy, wz = self.data.qvel[self._dofs]
        yaw = float(self.data.qpos[self._qpos[2]])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * vx + s * vy, -s * vx + c * vy, wz])

    def is_robot(self, body: int) -> bool:
        return bool(self.model.body(body).name.startswith(PREFIX))
