# from envs.envs_assistive.scratch_itch_envs import ScratchItchPR2MeshEnv, ScratchItchBaxterMeshEnv, ScratchItchSawyerMeshEnv, ScratchItchJacoMeshEnv, ScratchItchStretchMeshEnv, ScratchItchPandaMeshEnv
from envs.envs_assistive.drinking_envs import DrinkingPR2Env, DrinkingBaxterEnv, DrinkingSawyerEnv, DrinkingJacoEnv, DrinkingStretchEnv, DrinkingPandaEnv, DrinkingPR2HumanEnv, DrinkingBaxterHumanEnv, DrinkingSawyerHumanEnv, DrinkingJacoHumanEnv, DrinkingStretchHumanEnv, DrinkingPandaHumanEnv
from envs.envs_assistive.feeding_envs import FeedingPR2Env, FeedingBaxterEnv, FeedingSawyerEnv, FeedingJacoEnv, FeedingStretchEnv, FeedingPandaEnv, FeedingPR2HumanEnv, FeedingBaxterHumanEnv, FeedingSawyerHumanEnv, FeedingJacoHumanEnv, FeedingStretchHumanEnv, FeedingPandaHumanEnv
from envs.envs_assistive.feeding_envs import FeedingPR2MeshEnv, FeedingBaxterMeshEnv, FeedingSawyerMeshEnv, FeedingJacoMeshEnv, FeedingStretchMeshEnv, FeedingPandaMeshEnv
from envs.envs_assistive.human_testing import HumanTestingEnv
# from envs.envs_assistive.smplx_testing import SMPLXTestingEnv

from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)