from .scratch_itch import ScratchItchEnv
from .scratch_itch_mesh import ScratchItchMeshEnv
from envs.envs_assistive.agents import pr2, baxter, sawyer, jaco, stretch, panda, human, human_mesh
from envs.envs_assistive.agents.pr2 import PR2
from envs.envs_assistive.agents.baxter import Baxter
from envs.envs_assistive.agents.sawyer import Sawyer
from envs.envs_assistive.agents.jaco import Jaco
from envs.envs_assistive.agents.stretch import Stretch
from envs.envs_assistive.agents.panda import Panda
from envs.envs_assistive.agents.human import Human
from envs.envs_assistive.agents.human_mesh import HumanMesh
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class ScratchItchPR2Env(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchBaxterEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchSawyerEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchJacoEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchStretchEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchPandaEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchPR2HumanEnv(ScratchItchEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchPR2Human-v1', lambda config: ScratchItchPR2HumanEnv())

class ScratchItchBaxterHumanEnv(ScratchItchEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchBaxterHuman-v1', lambda config: ScratchItchBaxterHumanEnv())

class ScratchItchSawyerHumanEnv(ScratchItchEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchSawyerHuman-v1', lambda config: ScratchItchSawyerHumanEnv())


class ScratchItchJacoHumanEnv(ScratchItchEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchJacoHuman-v1', lambda config: ScratchItchJacoHumanEnv())


class ScratchItchStretchHumanEnv(ScratchItchEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchStretchHuman-v1', lambda config: ScratchItchStretchHumanEnv())

class ScratchItchPandaHumanEnv(ScratchItchEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchPandaHuman-v1', lambda config: ScratchItchPandaHumanEnv())

class ScratchItchPR2MeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchPR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=HumanMesh())

class ScratchItchBaxterMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchBaxterMeshEnv, self).__init__(robot=Baxter(robot_arm), human=HumanMesh())

class ScratchItchSawyerMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchSawyerMeshEnv, self).__init__(robot=Sawyer(robot_arm), human=HumanMesh())

class ScratchItchJacoMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchJacoMeshEnv, self).__init__(robot=Jaco(robot_arm), human=HumanMesh())

class ScratchItchStretchMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchStretchMeshEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=HumanMesh())

class ScratchItchPandaMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchPandaMeshEnv, self).__init__(robot=Panda(robot_arm), human=HumanMesh())

