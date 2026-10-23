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

"""The R1 Pro's URDF, read and posed; its swerve wheels; and the world it stands in."""

import math

import numpy as np
import pytest

from dim_r1_sim.config import CameraConfig, PhysicsConfig
from dim_r1_sim.robot import ROOT, load_urdf, r1pro_urdf, swerve, wheel_modules
from dim_r1_sim.world import R1World, compose


@pytest.fixture(scope="module")
def robot():
    return load_urdf(r1pro_urdf())


def test_every_link_hangs_off_base_link_with_its_meshes(robot) -> None:
    assert len(robot.links) == len(set(robot.links)) == 43
    poses = robot.link_poses()
    assert set(poses) == set(robot.links)
    for frame in ("lidar_chassis_left_link", "camera_head_left_link", "camera_head_right_link"):
        assert frame in poses
    assert all(v.mesh.is_file() for vs in robot.visuals.values() for v in vs)


def test_standing_pose_is_upright(robot) -> None:
    poses = robot.link_poses()
    head, left, right = (
        poses[f][:3, 3] for f in ("head_link", "camera_head_left_link", "camera_head_right_link")
    )
    assert head[2] > 1.2  # the torso stands straight up over the chassis
    assert left[1] - right[1] == pytest.approx(0.12, abs=0.01)  # the stereo baseline
    # Each eye's optical axis (z) looks forward and somewhat down.
    forward = poses["camera_head_left_link"][:3, 2]
    assert forward[0] > 0.8 and forward[2] < 0


def test_a_revolute_joint_turns_its_child(robot) -> None:
    bent = robot.link_poses({"torso_joint1": 0.5})
    assert bent["head_link"][0, 3] > robot.link_poses()["head_link"][0, 3] + 0.3


def test_swerve_steers_each_wheel_along_its_velocity(robot) -> None:
    modules = wheel_modules(robot)
    assert len(modules) == 3
    assert all(0.05 < radius < 0.1 for _, _, radius in modules)
    sideways = swerve(modules, np.array([0.0, 0.5, 0.0]), {})
    for steer, wheel, radius in modules:
        assert sideways[steer.name] == pytest.approx(math.pi / 2)
        assert sideways[wheel.name] == pytest.approx(0.5 / radius)
    backwards = swerve(modules, np.array([-0.5, 0.0, 0.0]), {})
    assert all(backwards[s.name] == pytest.approx(0.0, abs=1e-9) for s, _, _ in modules)
    assert all(backwards[w.name] < 0 for _, w, _ in modules)


def test_the_base_drives_and_stays_upright_on_bare_ground(robot) -> None:
    world = R1World(
        compose(None, robot, robot.link_poses(), PhysicsConfig(), CameraConfig()), PhysicsConfig()
    )
    world.reset(1.0, 2.0, math.pi / 2)
    for _ in range(100):  # 2 s at 0.5 m/s forward, facing +y
        world.tick(np.array([0.5, 0.0, 0.0]))
    position, rotation = world.base_pose()
    assert position[:2] == pytest.approx([1.0, 2.95], abs=0.05)
    assert position[2] == 0.0 and rotation[2, 2] == pytest.approx(1.0)
    assert world.model.body(world.base).name == "r1/" + ROOT
