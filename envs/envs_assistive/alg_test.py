import os
import numpy as np
from gym import spaces
import mujoco_py

from envs.gym_kuka_mujoco.envs.assets import kuka_asset_dir
from envs.gym_kuka_mujoco.utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat
from envs.mujoco.utils.kinematics import forwardKinSite, forwardKinJacobianSite, forwardVelKinSite
from envs.gym_kuka_mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, \
    get_joint_indices
from envs.mujoco.mujoco_config import MujocoConfig
from envs.gym_kuka_mujoco.utils.control_utils import *
from envs.gym_kuka_mujoco.utils.transform_utils import *

from collections.abc import Iterable
import transforms3d as transforms3d

state = np.array([1.57061e+00, 1.39156e-03, 1.56917e+00])
reference = np.array([-0.32692, 0.19711, -2.75531])

real_ori_attractor = transforms3d.euler.euler2mat(
    state[0],
    state[1],
    state[2],
    'sxyz')
ref_ori_attractor = transforms3d.euler.euler2mat(
    reference[0],
    reference[1],
    reference[2],
    'sxyz')

ori_error_point = orientation_error(real_ori_attractor, ref_ori_attractor)

mat_t = transforms3d.euler.euler2mat(state[0],
                                     state[1],
                                     state[2],
                                     'sxyz')
mat_d = transforms3d.euler.euler2mat(ori_error_point[0],
                                     ori_error_point[1],
                                     ori_error_point[2],
                                     'sxyz')
print("ori_error :",
      ori_error_point,
      orientation_error(mat_d, 2 * mat_t),
	  orientation_error(mat_d, mat_t),
      orientation_error(euler2mat(state), 2 * euler2mat(state))
      )
