import os
import random
import numpy as np

from envs.gym_kuka_mujoco.envs import kuka_env
from envs.gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from envs.gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from envs.gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from envs.gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat
from envs.gym_kuka_mujoco.envs.assets import kuka_asset_dir
from envs.gym_kuka_mujoco.utils.transform_utils import *


class PickPlaceEnv(kuka_env.KukaEnv):
    
    def __init__(self,
                 *args,
                 obs_scaling=0.1,
                 use_ft_sensor=False,
                 use_rel_pos_err=False,
                 pos_reward=True,
                 vel_reward=False,
                 **kwargs):
        
        # Store arguments.
        self.obs_scaling = obs_scaling
        self.use_ft_sensor = use_ft_sensor
        self.use_rel_pos_err = use_rel_pos_err
        self._max_episode_steps = 200
        
        self.context_low_bound = 20
        self.context_high_bound = 80
        
        # Resolve the models path based on the hole_id.
        kwargs['model_path'] = kwargs.get('model_path', 'pick_and_place.xml')
        super(PickPlaceEnv, self).__init__(*args, **kwargs)
        
        # Compute good states using inverse kinematics.
        # if self.random_target:
        #     raise NotImplementedError
        
        self.kuka_idx = [self.model.joint_name2id('robot0_right_j{}'.format(i)) for i in range(0, 7)]
        # self.nail_idx = self.model.joint_name2id('nail_position')
        self.init_qpos = np.zeros(9)
        self.init_qpos[self.kuka_idx] = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        
        print("model_nq :::", self.model.nq)
        print("model_nv :::", self.model.nv)
        self.pos_reward = pos_reward
        self.vel_reward = vel_reward

    def _get_reward(self, state, action):
        '''
            Compute single step reward.
        '''
        reward_info = dict()
        reward = 0.
        
        # if self.pos_reward:
        #     reward_info['nail_pos_reward'] = -self.data.qpos[self.nail_idx]
        #     reward += reward_info['nail_pos_reward']
        # if self.vel_reward:
        #     reward_info['nail_vel_reward'] = -self.data.qvel[self.nail_idx]
        #     reward += reward_info['nail_vel_reward']

        return reward*self.sac_reward_scale, reward_info  # *100 for SAC

    def _get_info(self):
        info = dict()
        target_pos, target_rot = forwardKinSite(self.sim, ['ee_site', 'target_ee_site'])
        info['distance'] = np.linalg.norm(target_pos[1] - target_pos[0])
        return info

    def _get_state_obs(self):
        '''
            Compute the observation at the current state.
        '''
        if not self.initialized:
            obs = np.zeros(16)
        else:
            # Return superclass observation.
            obs = super(PickPlaceEnv, self)._get_state_obs()

        # Return superclass observation stacked with the ft observation.
        if not self.initialized:
            ft_obs = np.zeros(6)
        else:
            # Compute F/T sensor data
            ft_obs = self.sim.data.sensordata
            obs = obs / self.obs_scaling

        if self.use_ft_sensor:
            obs = np.concatenate([obs, ft_obs])

        return obs

    def _get_target_obs(self):
        """
            Compute relative position error
        """
        if self.use_rel_pos_err:
            pos, rot = forwardKinSite(self.sim, ['ee_site', 'target_ee_site'])
            pos_obs = pos[0] - pos[1]
            quat_hammer_tip = mat2Quat(rot[0])
            quat_nail_top = mat2Quat(rot[1])
            rot_obs = subQuat(quat_hammer_tip, quat_nail_top)
            return np.concatenate([pos_obs, rot_obs])
        else:
            return np.array([self.data.qpos[self.nail_idx]])

    def _reset_state(self):
        '''
            Reset the robot state and return the observation.
        '''
        qpos = self.init_qpos.copy()
        qvel = np.zeros(9)
        self.set_state(qpos, qvel)

    def _reset_target(self):
        '''
            Resets the hole position
        '''
        if self.contextual_policy:
            hammer_friction = np.random.uniform(20, 80, 1)
            print("Initial friction ::::", hammer_friction)
            self.sim.model.dof_frictionloss[:] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        else:
            self.sim.model.dof_frictionloss[:] = np.array([0., 0., 0., 0., 0., 0., 0.])
        # raise NotImplementedError
    
    def get_context(self):
        """
            return frictionloss
        """
        if self.contextual_policy:
            self._reset_target()
            self.context = self.sim.model.dof_frictionloss[7]
            # context normalization :::
            self.context = (self.context - self.context_low_bound)/(self.context_high_bound - self.context_low_bound)
            return self.context
        else:
            exit("No contextual policy !!!")

    def set_waypoints(self):
        """ set waypoints """
        target_pos, target_rot = forwardKinSite(self.sim, ['ee_site', 'target_ee_site'])
        print("target_pos_0 :::", target_pos[0])
        print("target_pos_1 :::", target_pos[1])
        
        print("target_rot_0 :::", mat2euler(target_rot[0]))
        print("target_rot_1 :::", mat2euler(target_rot[1]))
    
        self.pos_optimal_point = target_pos[1]
        # self.quat_optimal_point = np.array([1.5708386, -1.3962705, 1.5707542]) + np.array([0.0, 0.0, 0.0])
        # self.quat_optimal_point = mat2Quat(target_rot[1])
        quat = mat2quat(euler2mat(np.array([1.5708386, -0.3, 1.57])))
        print("Initial quat :::", quat)
        self.quat_optimal_point = quat

        quat_hole_top = mat2Quat(target_rot[0])
        quat_hole_base = mat2Quat(np.array([1.5708386, -1.3962705, 1.0]))
        # quat_hole_top = np.array([1.5708386, -1.3962705, 1.5707542]) + np.array([0.0, 0.0, 0.0])
        # quat_hole_base = np.array([1.5708386, -1.3962705, 1.5707542]) + np.array([0.0, 0.0, 0.0])
    
        if self.controller.num_waypoints == 3:
            self.pos_list = np.concatenate(([target_pos[1]], [target_pos[2]], [target_pos[2] - [0.0, 0.0, -0.05]]),
                                           axis=0)
            self.quat_list = np.concatenate(([quat_hole_base], [quat_hole_top], [quat_hole_top]), axis=0)
        else:
            self.pos_list = np.concatenate(([target_pos[1]], [target_pos[1] - [0., 0., 0.1]]), axis=0)
            self.quat_list = np.concatenate(([quat_hole_base], [quat_hole_base]), axis=0)
    
        self.controller.set_waypoints(pos_list=self.pos_list,
                                      quat_list=self.quat_list,
                                      pos_optimal_point=self.pos_optimal_point,
                                      quat_optimal_point=self.quat_optimal_point
                                      )
        
        self.controller.set_target(np.array([0.1, 0.0, 1.2]), np.array([-0.00339245, -0.68766563, 0.72599291, -0.00622565]).astype(np.float64))
        
    def get_ee_pos(self):
        target_pos, target_rot = forwardKinSite(self.sim, ['ee_site', 'target_ee_site'])
        print("target_quat_1 :::", mat2Quat(target_rot[0]))
        print("target_quat_2 :::", mat2quat(target_rot[0]))
        
        print("target_pos_0 :::", target_pos[0])
        print("target_pos_1 :::", target_pos[1])
    
        print("target_rot_0 :::", mat2euler(target_rot[0]))
        print("target_rot_1 :::", mat2euler(target_rot[1]))
        
        return target_pos[0], mat2euler(target_rot[0])
