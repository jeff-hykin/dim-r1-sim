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

"""The chassis Mid-360, cast as dim-go2-sim casts its: the firing pattern and range response of
dim_go2_sim.mid360, each 5 ms slice of a scan cast from where the sensor is then.

The office's drawn surfaces (not its coarse collision hulls) go through Embree once, through
dim-go2-sim's Raycaster. The R1's own body (torso, arms, wheels; a million triangles) is rigid on
its base, so it goes through Embree too, in base_link's frame, each ray turned into it: MuJoCo's
ray casts against those meshes would take 15x real time.
"""

from __future__ import annotations

from collections.abc import Callable

from dim_go2_sim import config as go2_config
from dim_go2_sim.lidar import LidarScan, Raycaster, RollingLidar, static_geoms, static_triangles
import mujoco
import numpy as np
from numpy.typing import NDArray
import open3d as o3d  # type: ignore[import-untyped]

from dim_r1_sim.config import LidarConfig
from dim_r1_sim.robot import Robot, Transform, read_stl, transformed

__all__ = ["BodyRaycaster", "LidarScan", "chassis_lidar", "office_raycaster"]

Pose = tuple[NDArray[np.float64], NDArray[np.float64]]  # position, rotation matrix


def office_raycaster(world: mujoco.MjModel, data: mujoco.MjData) -> Raycaster:
    """Casts against the office's drawn geoms and floor, not its collision hulls (group 3)."""
    bodies = {b for b in range(world.nbody) if world.body(b).name.startswith("r1/")}
    surfaces = static_geoms(world, bodies) & (world.geom_group != 3)
    return Raycaster(world, data, bodies, None, 0, static_triangles(world, data, surfaces))


class BodyRaycaster:
    """The office, and the robot's own meshes posed at *poses* on a base at *base_pose()*; nearest
    hit wins. *skip* links are left out (the lidar's own housing)."""

    def __init__(
        self,
        office: Raycaster,
        robot: Robot,
        poses: dict[str, Transform],
        base_pose: Callable[[], Pose],
        skip: tuple[str, ...] = (),
    ) -> None:
        self._office = office
        self._base_pose = base_pose
        corners = [
            transformed(read_stl(v.mesh).reshape(-1, 3), poses[link] @ v.origin)
            for link, visuals in robot.visuals.items()
            if link not in skip
            for v in visuals
        ]
        vertices = np.concatenate(corners)
        self._body = o3d.t.geometry.RaycastingScene()
        self._body.add_triangles(
            o3d.core.Tensor(vertices),
            o3d.core.Tensor(np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)),
        )

    def cast(
        self, origin: NDArray[np.float64], directions: NDArray[np.float64], max_range: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Distance and hit normal (facing the ray) per world-frame unit direction; -1 is a miss."""
        dist, normals = self._office.cast(origin, directions, max_range)
        position, rotation = self._base_pose()
        rays = np.empty((len(directions), 6), np.float32)
        rays[:, :3] = rotation.T @ (np.asarray(origin) - position)
        rays[:, 3:] = directions @ rotation
        hit = self._body.cast_rays(o3d.core.Tensor(rays))
        own = hit["t_hit"].numpy().astype(np.float64)
        nearer = np.isfinite(own) & (own <= max_range) & ((dist < 0) | (own < dist))
        dist[nearer] = own[nearer]
        own_normals = hit["primitive_normals"].numpy().astype(np.float64)[nearer] @ rotation.T
        away = np.einsum("ij,ij->i", own_normals, directions[nearer]) > 0
        own_normals[away] *= -1
        normals[nearer] = own_normals
        return dist, normals


def chassis_lidar(
    config: LidarConfig,
    raycaster: Raycaster | BodyRaycaster,
    sensor_pose: Callable[[], Pose],
) -> RollingLidar:
    device = go2_config.LidarConfig(**config.model_dump(exclude={"enabled", "frame"}))
    return RollingLidar(device, raycaster, sensor_pose)  # type: ignore[arg-type]
