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

"""The R1 Pro and the office as one URDF, shared as dimOS's UrdfManager shares one (dim-controller).

As dim-go2-sim's: the XML goes out on robot_description, each mesh named
package://dim_r1_sim/<path>?hash=<sha256 of the file>, and a file_request for one of those names is
answered on file_info with the file, unless the request carries &have=<that hash>.

The root is world; base_link hangs off it by a floating joint and every link below it keeps its
URDF joint (each posed by tf). The robot's meshes are Galaxea's STL files, sent as they are. The
office is fixed at world as <group>scene</group> link scene_mesh: its drawn geoms under the
ground storey's ceiling, one visual per colour, in STL tiles built once into the cache.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path
import xml.etree.ElementTree as ET

from dim_go2_sim.lidar import static_triangles
from dim_go2_sim.urdf import (
    MeshFile,
    SharedUrdf,
    _cached,
    _digest,
    _file_digest,
    _grouped_link,
    stl,
    tiles,
)
import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dim_r1_sim.config import UrdfConfig
from dim_r1_sim.office import cache_root
from dim_r1_sim.robot import ROOT, Robot, Transform

__all__ = ["MeshFile", "SharedUrdf", "office_colours", "share_urdf"]

PACKAGE = "dim_r1_sim"
# Bump when the files change, so cached ones are rebuilt.
ASSETS_VERSION = 1


def share_urdf(
    robot: Robot,
    world: mujoco.MjModel | None,
    world_data: mujoco.MjData | None,
    office_key: str,
    config: UrdfConfig,
    prefix: str = "",
    cache_dir: Path | None = None,
) -> SharedUrdf:
    """The URDF of *robot* and the office in *world* (None: the robot alone), each link named as
    its tf frame (after *prefix*), the office's files built once into the cache."""
    files: dict[str, Path] = {}
    xml = ET.Element("robot", name="r1pro")
    world_link = prefix + "world"
    ET.SubElement(xml, "link", name=world_link)
    for link in robot.links:
        element = ET.SubElement(xml, "link", name=prefix + link)
        for i, visual in enumerate(robot.visuals[link]):
            uri = f"package://{PACKAGE}/r1pro/{visual.mesh.name}"
            files[uri] = visual.mesh
            node = ET.SubElement(element, "visual")
            ET.SubElement(node, "origin", **_origin(visual.origin))
            ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=uri)
            material = ET.SubElement(node, "material", name=f"{link}_{i}")
            ET.SubElement(material, "color", rgba=_numbers(visual.rgba))
    joint = ET.SubElement(xml, "joint", name="base_joint", type="floating")
    ET.SubElement(joint, "parent", link=world_link)
    ET.SubElement(joint, "child", link=prefix + ROOT)
    for j in robot.joints:
        element = ET.SubElement(xml, "joint", name=j.name, type=j.type)
        ET.SubElement(element, "origin", **_origin(j.origin))
        ET.SubElement(element, "parent", link=prefix + j.parent)
        ET.SubElement(element, "child", link=prefix + j.child)
        if j.type != "fixed":
            ET.SubElement(element, "axis", xyz=_numbers(j.axis))
        if j.limits is not None:
            lower, upper = (f"{limit:.6g}" for limit in j.limits)
            ET.SubElement(element, "limit", lower=lower, upper=upper, effort="0", velocity="0")
    if world is not None and world_data is not None:
        settings = f"{office_key} {config.scene_top_z} {config.tile_faces}"
        tag = f"{ASSETS_VERSION} {version('dim-r1-sim')} {settings}"
        folder = _cached(
            cache_root(cache_dir) / "urdf" / f"office_555-{_digest(tag)}",
            lambda out: _write_office(world, world_data, config, out),
        )
        link = _grouped_link(xml, prefix + "scene_mesh", "scene")
        for path in sorted(folder.glob("colour_*")):
            rgba = (path / "rgba").read_text()
            for tile in sorted(path.glob("mesh_*.stl")):
                uri = f"package://{PACKAGE}/scenes/office_555/{path.name}/{tile.name}"
                files[uri] = tile
                node = ET.SubElement(link, "visual")
                ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=uri)
                material = ET.SubElement(node, "material", name=f"scene_{path.name}")
                ET.SubElement(material, "color", rgba=rgba)
        joint = ET.SubElement(xml, "joint", name="scene_mesh_joint", type="fixed")
        ET.SubElement(joint, "parent", link=world_link)
        ET.SubElement(joint, "child", link=prefix + "scene_mesh")
    hashed = {name: (path, _file_digest(path)) for name, path in files.items()}
    for mesh in xml.iter("mesh"):
        name = mesh.attrib["filename"]
        mesh.set("filename", f"{name}?hash={hashed[name][1]}")
    return SharedUrdf(ET.tostring(xml, encoding="unicode"), hashed)


def office_colours(
    world: mujoco.MjModel, data: mujoco.MjData, top_z: float
) -> dict[tuple[float, ...], NDArray[np.bool_]]:
    """The office's drawn geoms wholly below *top_z*, by colour (its material's, a builtin
    texture's mean times it, or the geom's own): what the camera sees, but the robot and the
    collision hulls (group 3) and the floor plane, which a viewer draws its own of."""
    drawn = (world.body_weldid[world.geom_bodyid] == 0) & (world.geom_group != 3)
    drawn &= world.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    drawn &= world.geom_rgba[:, 3] > 0
    tops = data.geom_xpos[:, 2] + world.geom_rbound
    by_colour: dict[tuple[float, ...], NDArray[np.bool_]] = {}
    for geom in np.flatnonzero(drawn):
        if tops[geom] > top_z and _top(world, data, geom) > top_z:
            continue
        rgba = _colour(world, geom)
        if rgba[3] < 0.05:
            continue
        mask = by_colour.setdefault(rgba, np.zeros(world.ngeom, bool))
        mask[geom] = True
    return by_colour


def _top(world: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
    """The geom's highest point, as the triangles it is shared as."""
    only = np.zeros(world.ngeom, bool)
    only[geom] = True
    vertices, _ = static_triangles(world, data, only)
    return float(vertices[:, 2].max()) if len(vertices) else -np.inf


def _colour(world: mujoco.MjModel, geom: int) -> tuple[float, ...]:
    material = world.geom_matid[geom]
    if material < 0:
        rgba = world.geom_rgba[geom].astype(np.float64)
    else:
        rgba = world.mat_rgba[material].astype(np.float64)
        texture = world.mat_texid[material][mujoco.mjtTextureRole.mjTEXROLE_RGB]
        if texture >= 0:
            start = world.tex_adr[texture]
            size = (
                world.tex_height[texture] * world.tex_width[texture] * world.tex_nchannel[texture]
            )
            pixels = world.tex_data[start : start + size].reshape(-1, world.tex_nchannel[texture])
            rgba[:3] *= pixels[:, :3].mean(0) / 255
    return tuple(round(float(c), 3) for c in rgba)


def _write_office(
    world: mujoco.MjModel, data: mujoco.MjData, config: UrdfConfig, folder: Path
) -> None:
    for i, (rgba, geoms) in enumerate(office_colours(world, data, config.scene_top_z).items()):
        out = folder / f"colour_{i:03d}"
        out.mkdir()
        (out / "rgba").write_text(_numbers(rgba))
        vertices, faces = static_triangles(world, data, geoms)
        for t, tile in enumerate(tiles(vertices, faces, config.tile_faces)):
            (out / f"mesh_{t:03d}.stl").write_bytes(stl(*tile))


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _origin(pose: Transform) -> dict[str, str]:
    return {
        "xyz": _numbers(pose[:3, 3]),
        "rpy": _numbers(Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")),
    }
