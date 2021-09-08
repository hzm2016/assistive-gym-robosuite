from envs.robosuite.robosuite.environments.base import make

# Manipulation environments
from envs.robosuite.robosuite.environments.manipulation.lift import Lift
from envs.robosuite.robosuite.environments.manipulation.stack import Stack
from envs.robosuite.robosuite.environments.manipulation.extraction import Extract
from envs.robosuite.robosuite.environments.manipulation.nut_assembly import NutAssembly
from envs.robosuite.robosuite.environments.manipulation.pick_place import PickPlace
from envs.robosuite.robosuite.environments.manipulation.door import Door
from envs.robosuite.robosuite.environments.manipulation.wipe import Wipe
from envs.robosuite.robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from envs.robosuite.robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from envs.robosuite.robosuite.environments.manipulation.two_arm_handover import TwoArmHandover

from envs.robosuite.robosuite.environments import ALL_ENVIRONMENTS
from envs.robosuite.robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from envs.robosuite.robosuite.robots import ALL_ROBOTS
from envs.robosuite.robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.2.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
