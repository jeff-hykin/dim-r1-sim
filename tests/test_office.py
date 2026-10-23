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

"""The office the R1 stands in, its share in robot_description, and the lidar in it."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dim_r1_sim.config import CameraConfig, LidarConfig, PhysicsConfig, UrdfConfig
from dim_r1_sim.lidar import BodyRaycaster, chassis_lidar, office_raycaster
from dim_r1_sim.office import office_dir
from dim_r1_sim.robot import load_urdf, r1pro_urdf, read_stl
from dim_r1_sim.urdf import share_urdf
from dim_r1_sim.world import R1World, compose


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    robot = load_urdf(r1pro_urdf())
    poses = robot.link_poses()
    office = office_dir(cache_dir=tmp_path_factory.getbasetemp() / "cache")
    world = R1World(compose(office, robot, poses, PhysicsConfig(), CameraConfig()), PhysicsConfig())
    world.reset(0.0, 0.0, 0.0)
    return robot, poses, world, tmp_path_factory.getbasetemp() / "cache"


def test_the_robot_spawns_clear_of_the_office(scene) -> None:
    _, _, world, _ = scene
    assert world.model.ngeom > 3900  # office_555's geoms and the R1's
    assert world.data.ncon == 0
    for _ in range(25):
        world.tick(np.zeros(3))
    assert world.base_pose()[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_robot_description_names_every_link_and_file(scene) -> None:
    robot, _, world, cache = scene
    shared = share_urdf(robot, world.model, world.data, "test", UrdfConfig(), cache_dir=cache)
    root = ET.fromstring(shared.xml)
    links = {link.attrib["name"] for link in root.iter("link")}
    assert links == {"world", "scene_mesh", *robot.links}
    scene_link = next(link for link in root.iter("link") if link.attrib["name"] == "scene_mesh")
    assert scene_link.find("group").text == "scene"
    assert len(scene_link.findall("visual")) > 10  # the office, colour by colour
    for mesh in root.iter("mesh"):
        name, _, query = mesh.attrib["filename"].partition("?hash=")
        assert shared.files[name][1] == query
    # The roof is left out: no shared triangle is above the ground storey's ceiling.
    for name, (path, _) in shared.files.items():
        if "/scenes/" in name:
            assert read_stl(path)[..., 2].max() <= UrdfConfig().scene_top_z + 1e-3
    first = next(iter(shared.files))
    assert shared.answer(first).data == shared.files[first][0].read_bytes()
    assert shared.answer(f"{first}?have={shared.files[first][1]}") is None


def test_the_lidar_sees_the_office_and_the_robot_body(scene) -> None:
    robot, poses, world, _ = scene
    config = LidarConfig()
    mount = poses[config.frame]

    def pose():
        position, rotation = world.base_pose()
        return position + rotation @ mount[:3, 3], rotation @ mount[:3, :3]

    raycaster = BodyRaycaster(
        office_raycaster(world.model, world.data), robot, poses, world.base_pose, (config.frame,)
    )
    lidar = chassis_lidar(config, raycaster, pose)
    scans = []
    while not scans:
        world.tick(
            np.zeros(3), lambda: scans.append(s) if (s := lidar.advance(world.time)) else None
        )
    ranges = np.linalg.norm(scans[0].points, axis=1)
    assert len(ranges) > 10_000
    assert (ranges < 1.0).mean() > 0.02  # its own torso and arms, beside the chassis lidar
    assert (ranges > 5.0).mean() > 0.5  # the office around it
