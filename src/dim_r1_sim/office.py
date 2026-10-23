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

"""office_555, the default room of dimOS's MuJoCo sim: found or downloaded, and loaded as an MjSpec.

dimOS keeps it in git LFS (data/.lfs/mujoco_office_555.tar.gz, on its own LFS server); a
pip-installed dimos has no data, so the archive is fetched by its LFS object id, checked, and
unpacked once.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.request
import xml.etree.ElementTree as ET

import mujoco

NAME = "mujoco_office_555"
SCENE_XML = "scene_office_555.xml"
# The archive as dimOS commit 1c5e5add ("feat(sim): the 555 office as the default MuJoCo") ships it.
LFS_BATCH = "https://lfs.dimensionalos.com/dimensionalOS/dimos/objects/batch"
LFS_OID = "557f52ae7d224fc250f42957035a60a3921ad1c4ea22ed2e714bc49e21ba30af"
LFS_SIZE = 1_167_195
_LFS_JSON = "application/vnd.git-lfs+json"


def cache_root(cache_dir: Path | None) -> Path:
    return (
        cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    ) / "dim_r1_sim"


def office_dir(override: Path | None = None, cache_dir: Path | None = None) -> Path:
    """office_555's folder: *override*, else the cached download, fetched the first time."""
    if override is not None:
        if not (override / SCENE_XML).is_file():
            raise FileNotFoundError(f"{override} has no {SCENE_XML}")
        return override
    folder = cache_root(cache_dir) / NAME
    if (folder / SCENE_XML).is_file():
        return folder
    archive = _download()
    partial = folder.with_name(f"{folder.name}.{os.getpid()}.tmp")
    partial.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(partial, filter="data")
    (partial / NAME).rename(folder)
    partial.rmdir()
    return folder


def _download() -> bytes:
    """The archive's bytes, by its LFS object id (the batch API hands out a short-lived URL)."""
    request = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": LFS_OID, "size": LFS_SIZE}],
        }
    ).encode()
    headers = {"Accept": _LFS_JSON, "Content-Type": _LFS_JSON}
    with urllib.request.urlopen(urllib.request.Request(LFS_BATCH, request, headers)) as reply:
        href = json.load(reply)["objects"][0]["actions"]["download"]["href"]
    with urllib.request.urlopen(href) as reply:
        data: bytes = reply.read()
    if hashlib.sha256(data).hexdigest() != LFS_OID:
        raise RuntimeError(f"{NAME}: the download is not LFS object {LFS_OID}")
    return data


def office_xml(folder: Path) -> str:
    """The scene's MJCF with its meshes and textures by absolute path, as dimOS loads it."""
    root = ET.parse(folder / SCENE_XML).getroot()
    for element, subfolder in (("mesh", "meshes"), ("texture", "textures")):
        for asset in root.iter(element):
            if file := asset.get("file"):
                asset.set("file", (folder / subfolder / file).as_posix())
    return ET.tostring(root, encoding="unicode")


def office_spec(folder: Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(office_xml(folder))
