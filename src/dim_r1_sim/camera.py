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

"""The head's two colour cameras, drawn by MuJoCo from the world as it stands (RGB, no depth).

office_555 is drawn by its own visual geoms and materials, as dimOS's MuJoCo sim draws it, and the
R1's own meshes with it: an eye sees its arms when they are in view. The renderer's GL context
belongs to the thread that made it, so a HeadCameras is made and used on one thread. Headless
Linux needs MUJOCO_GL=egl (or osmesa).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from numpy.typing import NDArray

from dim_r1_sim.config import CameraConfig
from dim_r1_sim.world import EYES, PREFIX


def focal_length(config: CameraConfig) -> float:
    return config.height / (2 * math.tan(math.radians(config.fovy_deg) / 2))


class HeadCameras:
    def __init__(self, model: mujoco.MjModel, config: CameraConfig) -> None:
        self._renderer = mujoco.Renderer(model, config.height, config.width)
        self._cameras = {eye: model.camera(f"{PREFIX}head_{eye}").id for eye in EYES}

    def capture(self, data: mujoco.MjData, eye: str) -> NDArray[np.uint8]:
        """The eye's RGB image, (height, width, 3), of *data* as it is now."""
        self._renderer.update_scene(data, self._cameras[eye])
        image: NDArray[np.uint8] = self._renderer.render()
        return image

    def close(self) -> None:
        self._renderer.close()
