"""The simulated R1 Pro standing in dimOS's office, on zenoh: drive it with cmd_vel."""

import sys

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.module_coordinator import ModuleCoordinator

from dim_r1_sim import R1Sim

drive = R1Sim.blueprint().global_config(transport="zenoh")

if __name__ == "__main__":
    ModuleCoordinator.build(drive, BlueprintConfigParser(drive).parse(sys.argv[1:])).loop()
