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

"""A simulated Galaxea R1 Pro for dimOS, standing in dimOS's office: cmd_vel in; chassis Mid-360,
head cameras, odometry and tf out."""

from dim_r1_sim.config import (
    CameraConfig,
    LidarConfig,
    PhysicsConfig,
    RobotConfig,
    SceneConfig,
    UrdfConfig,
)
from dim_r1_sim.module import R1Sim, R1SimConfig

__all__ = [
    "CameraConfig",
    "LidarConfig",
    "PhysicsConfig",
    "R1Sim",
    "R1SimConfig",
    "RobotConfig",
    "SceneConfig",
    "UrdfConfig",
]
