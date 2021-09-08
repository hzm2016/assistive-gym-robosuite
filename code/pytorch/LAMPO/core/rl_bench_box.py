import numpy as np
np.set_printoptions(precision=5)
import matplotlib.patches as patches
import time
import os

from envs.gym_kuka_mujoco import kuka_asset_dir
import mujoco_py
from mujoco_py.generated import const
from envs.gym_kuka_mujoco.controllers import iMOGVIC
from envs.gym_kuka_mujoco.utils.transform_utils import *
from envs.robosuite.robosuite.utils import transform_utils as trans

from code.pytorch.LAMPO.core.task_interface import TaskInterface
from code.pytorch.LAMPO.core.rrt_star import RRTStar

from romi.movement_primitives import ClassicSpace, MovementPrimitive, LearnTrajectory
from romi.groups import Group
from romi.trajectory import NamedTrajectory, LoadTrajectory

import transforms3d as transforms3d

from envs.robosuite.robosuite.controllers import *
from gym import spaces

import pybullet as p

from tensorboardX import SummaryWriter as FileWriter
import imageio


def render_frame(viewer, pos, euler):
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])
    
    # mat = quat2mat(quat)
    mat = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2], 'sxyz')
    cylinder_half_height = 0.02
    pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[.005, .005, cylinder_half_height],
                      mat=mat)
    
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[cylinder_half_height, .005, .005],
                      mat=mat)
    
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[.005, cylinder_half_height, .005],
                      mat=mat)


class AssistiveOnePointEnv(TaskInterface):
    """
        Interface with MujocoEnv ::: iMOGIC
    """
    
    def __init__(self, env, set_params):
        super(AssistiveOnePointEnv).__init__()
        self._env = env
        self.mujoco_model = Mujoco_model(set_params["controller_options"], **set_params["mujoco_options"])
        self._env_name = set_params["env"]
        
        self.total_timesteps = 0
        self.episode_timesteps = 0
        self.episode_number = 0
        self.episode_reward = 0
        self.reward_scale = 0.01
        
        self.render = set_params["env_options"]["render"]
        
        self.max_episode_steps = set_params["alg_options"]["max_episode_steps"]
        self._state_dim = set_params["alg_options"]['context_dim']
        self._context_dim = set_params["alg_options"]['context_dim']
        self._latent_parameter_dim = set_params["alg_options"]['parameter_dim']
        self._latent_parameter_high = set_params["controller_options"]['stiffness_high']
        self._latent_parameter_low = set_params["controller_options"]['stiffness_low']
        self.num_way_points = set_params["controller_options"]['num_waypoints']
        self.action_dim = set_params["alg_options"]['action_dim']

        self.tool_delta_pos = np.array(set_params["env_options"]["tool_delta_pos"][self._env_name])
        self.tool_delta_ori = np.array(set_params["env_options"]["tool_delta_ori"][self._env_name])
        
        self.context = None
        self.target_pos = None
        self.target_ori = None
        self.target_euler = None
        
        self.delta_pose_high = self.mujoco_model.delta_pose_high
        self.delta_pose_low = self.mujoco_model.delta_pose_low
        self.delta_pose_scale = self.mujoco_model.delta_pose_scale

        self.delta_pose_params = np.zeros(self.action_dim)
        
        self.target_pos_list = np.zeros((self.mujoco_model.num_way_points, 3))
        self.target_euler_list = np.zeros((self.mujoco_model.num_way_points, 3))
    
    def reset(self):
        """
            get initial context
        """
        print("Environment Reset !!!")
        print("+" * 50)
        if self.render:
            self._env.render()
        
        observation = self._env.reset()
        # print('robot_joint_angles:', observation['robot_joint_angles'])
        
        # reset joints and way points
        self.mujoco_ee_pose = self.mujoco_model.reset(observation['robot_joint_angles'])
        
        # self.initial_tool_pos, self.initial_tool_orient = self._env.get_tool_pose()
        # self.initial_tool_euler = transforms3d.euler.quat2euler(self.initial_tool_orient, 'sxyz')
        
        self.get_ee_state()
        
        self.target_pos_list[0] = self.mujoco_target_pos
        # self.target_pos_list[1] = self.mujoco_target_pos_1
        
        self.target_euler_list[0] = self.mujoco_target_euler
        # self.target_euler_list[1] = self.mujoco_target_euler_1
        
        self.set_waypoints(way_points_list=1,
                           target_pos=self.target_pos_list,
                           target_euler=self.target_euler_list)
        
        print("mujoco_target_pos :", self.mujoco_target_pos,
              "mujoco_target_euler :", np.array(self.mujoco_target_euler))
        
        self.read_context()
        return observation
    
    def get_ee_state(self):
        self.tool_pos, self.tool_orient = self._env.get_tool_pose()
        self.tool_euler = transforms3d.euler.quat2euler(self.tool_orient, 'sxyz')
        # print("tool_pos :", self.tool_pos, "tool_euler :", np.array(self.tool_euler),
        #       np.array(p.getEulerFromQuaternion(np.array(self.tool_orient), physicsClientId=self._env.id)))
        
        self.ee_pos, self.ee_ori = self._env.robot.get_ee_pose()
        # print("original ori :", self.ee_ori)
        self.ee_euler = transforms3d.euler.quat2euler(self.ee_ori, 'sxyz')
        # print("ee_pos :", self.ee_pos, "ee_euler :", np.array(self.ee_euler),
        #       np.array(p.getEulerFromQuaternion(np.array(self.ee_ori), physicsClientId=self._env.id)))
        
        self.mujoco_model_ee = self.mujoco_model.get_ee_pose()
        # print("mujoco_model_ee :", np.array(self.mujoco_model_ee))
        
        self.target_pos, self.target_ori, \
        self.target_pos_ref, self.target_orient_ref, \
        self.target_pos_tool, self.target_orient_tool = self._env.get_context()
        
        self.target_euler_ref = transforms3d.euler.quat2euler(self.target_orient_ref, 'sxyz')
        self.target_euler = transforms3d.euler.quat2euler(self.target_ori, 'sxyz')
        
        # print("target_pos :", self.target_pos, "target_euler :", np.array(self.target_euler))
        # print("reference_target_pos :", self.target_pos_ref,
        #       "reference_target_euler :", self.target_euler_ref)
        
        # self.delta_pos = self.target_pos_ref + self.tool_delta_pos - self.tool_pos
        # self.target_pos_ref_new, self.target_orient_ref_new = p.multiplyTransforms(
        #     self.target_pos_ref, self.target_orient_ref,
        #     self.ee_pos - self.tool_pos, [0, 0, 0, 1], physicsClientId=self._env.id)
        
        # self.tool_delta_pos, desired_ori = p.multiplyTransforms(
        #     self.ee_pos, self.ee_ori,
        #     self.ee_pos - self.tool_pos, [0, 0, 0, 1], physicsClientId=self._env.id)
        
        # + self.tool_delta_pos
        # print("tool_delta_pos :", self.tool_delta_pos, np.array(p.getEulerFromQuaternion(np.array(desired_ori), physicsClientId=self._env.id)))
        
        # self.initial_tool_delta_ori = np.array([-self.target_euler_ref[0], -self.target_euler_ref[1], self.target_euler_ref[2] - 1.57])
        # self.initial_tool_delta_ori = np.array([self.target_euler_ref[1], self.target_euler_ref[0], 0.0])
        
        self.pybullet_ori_euler = np.array(
            p.getEulerFromQuaternion(np.array(self.target_orient_tool), physicsClientId=self._env.id))
        # print("pybullet_euler :", self.pybullet_ori_euler)
        
        # self.delta_pos = self.target_pos_tool - self.ee_pos + self.tool_delta_pos
        self.delta_pos = self.target_pos_tool - self.ee_pos + self.delta_pose_params[:3]
        
        # self.delta_pos = self.target_pos_ref + (self.tool_delta_pos - self.ee_pos) - self.ee_pos
        self.mujoco_target_pos = self.mujoco_ee_pose[:3] + self.delta_pos
        
        # self.mujoco_target_euler = \
        #     self._desired_ori(self.mujoco_ee_pose[3:],
        #                       self.pybullet_ori_euler + self.tool_delta_ori)
        self.mujoco_target_euler = \
            self._desired_ori(self.mujoco_ee_pose[3:],
                              self.pybullet_ori_euler + self.delta_pose_params[3:])
        
        # self.delta_pos_1 = self.target_pos_ref - self.ee_pos + self.tool_delta_pos
        #
        # # self.delta_pos = self.target_pos_ref + (self.tool_delta_pos - self.ee_pos) - self.ee_pos
        # self.mujoco_target_pos_1 = self.mujoco_ee_pose[:3] + self.delta_pos_1
        # self.mujoco_target_euler_1 = self._desired_ori(self.mujoco_ee_pose[3:], np.array([0.0, 0.0, 0.0]))
    
    def read_context(self):
        """
            return context and target
        """
        self.target_pos, self.target_ori, _, _, _, _ = self._env.get_context()
        self.target_euler = transforms3d.euler.quat2euler(self.target_ori, 'sxyz')
        # self.context = np.concatenate((self.target_pos, self.target_ori), axis=0)
        self.context = np.concatenate((self.delta_pos, self.target_ori), axis=0)
        return self.context
    
    def send_movement(self, params):
        """
            send parameters
        """
        self.send_params(params)
        
        reward, context, info = self.run_single_trajectory(n=1)
        self.get_ee_state()
        
        # print("error_pos :", self.tool_pos - self.initial_spoon_pos)
        # print("Final dist state :", self.mujoco_model.get_state_dist())
        return reward, context, info
    
    def set_waypoints(self, way_points_list=None, target_pos=None, target_euler=None):
        """
            set way points
        """
        # print("Mujoco_target_pos :", target_pos, 'target_euler :', target_euler)
        if way_points_list is None:
            self.target_pos_list = np.tile(target_pos, (self.mujoco_model.num_way_points, 1))
            self.target_euler_list = np.tile(target_euler, (self.mujoco_model.num_way_points, 1))
        
        self.mujoco_model.set_waypoints(self.target_pos_list, self.target_euler_list)
    
    def send_params(self, params):
        """
            set relations
        """
        self.mujoco_model.set_impedance_params(params[:self.num_way_points * self.action_dim])
        # print("params :", params[-self.action_dim:])
        # self.delta_pose_params = params[-self.action_dim:] * self.delta_pose_scale[-self.action_dim:] \
        #                          + self.delta_pose_low[-self.action_dim:]
        # print("delta_pose_params :", self.delta_pose_params)
    
    def run_single_trajectory(self, n=None):
        average_reward = 0.0
        info = None
        context = None
        self.done = False
        self.episode_number += 1
        self.episode_timesteps = 0
        self.episode_reward = 0.0
        for i in range(n):
            obs = self.reset()
            context = self.read_context()
            joint_last = obs['robot_joint_angles'].copy()
            while True:
                robot_action = np.zeros(6)
                joint = self.mujoco_model.step(robot_action)
                
                # control human or not
                human_action = np.zeros(self._env.action_human_len)
                
                action = {'robot': joint.copy() - joint_last, 'human': human_action}
                # env.action_space_human.sample()
                # joint_list.append(joint[0].copy())
                # print("action :", action)
                np.putmask(action['robot'], action['robot'] > 3.14, 0.0)
                np.putmask(action['robot'], action['robot'] < -3.14, 0.0)
                
                obs, reward, self.done, info = self._env.step(action)
                # obs, reward, self.done, info = self._env.step(action, joint)
                
                # print("done robot :", self.done['robot'])
                # print("done human :", self.done['human'])
                joint_last = obs['robot_joint_angles'].copy()
                # print("joint last :", joint_last)
                
                # done_bool = 0 if self.episode_timesteps - 1 == self.max_episode_steps else float(self.done['robot'])
                
                if self.render:
                    self._env.render()
                
                self.episode_reward += reward['robot']
                # self.episode_reward += reward['human']
                
                self.episode_timesteps += 1
                self.total_timesteps += 1
                
                if self.mujoco_model.get_state_dist() < 0.006:
                    break
                # print("mujoco_dist :", self.mujoco_model.get_state_dist())
                
                if self.done['__all__'] or info['robot']['task_success']:
                    break
            
            print("episode_reward :", self.episode_reward)
            
            # if self.done or self.episode_timesteps == self.max_episode_steps - 1:
            #     average_reward += self.episode_reward
            # else:
            #     average_reward += -300.0
            
            average_reward += self.episode_reward
            if info['robot']['task_success']:
                average_reward += 100
            else:
                average_reward += -100.0
            
            # print('joint :', joint.copy())
            # print('final :', obs['robot_joint_angles'].copy())
            print("Success :", info['robot']['task_success'], "Episode timesteps :", self.episode_timesteps,
                  "Reward :", np.around(average_reward / n, 4) * self.reward_scale)
            
            self.episode_number += 1
            self.episode_timesteps = 0
            self.episode_reward = 0.0
        
        return np.around(average_reward / n, 4) * self.reward_scale, context, info['robot']
    
    def get_demonstrations(self, num_traj=50):
        """
            generate demonstration samples
        """
        params_list = np.random.uniform(0, 1, size=(num_traj, self._latent_parameter_dim)) \
                      * (np.array(self._latent_parameter_high) - np.array(self._latent_parameter_low)) \
                      + np.array(self._latent_parameter_low)
        
        # # params_list = np.random.uniform(0, 1, size=(num_traj, self._env.latent_parameter_dim))
        print("params_list :", params_list)
        
        reward_list = []
        context_list = []
        for i in range(num_traj):
            # context, _, _, _ = self._env.get_context()
            params = params_list[i, :]
            reward, context, info = self.send_movement(params)
            context_list.append(context.tolist())
            # print("info :", info)
            reward_list.append(np.copy(reward))
        return np.hstack((np.array(context_list), params_list)), np.array(reward_list)
        # return np.hstack((np.array(context_list), params_list)), reward_list
        # return [np.concatenate((np.array(context_list), params_list), axis=0)]
    
    def get_context_dim(self):
        return self._context_dim
    
    def get_impedance_dim(self):
        return self._latent_parameter_dim
    
    def _desired_ori(self, current_euler, rot_euler):
        # convert axis-angle value to rotation matrix
        # quat_error = trans.axisangle2quat(rot_euler)
        # rotation_mat_error = trans.quat2mat(quat_error)
        # curr_mat = trans.euler2mat(current_euler)
        rotation_mat_error = transforms3d.euler.euler2mat(rot_euler[0], rot_euler[1], rot_euler[2], 'sxyz')
        # rotation_mat_error = trans.quat2mat(quat_error)
        # curr_mat = trans.euler2mat(current_euler)
        curr_mat = transforms3d.euler.euler2mat(current_euler[0], current_euler[1], current_euler[2], 'sxyz')
        goal_orientation = np.dot(rotation_mat_error, curr_mat)
        # return trans.mat2euler(goal_orientation, 'sxyz')
        return transforms3d.euler.mat2euler(goal_orientation, 'sxyz')


class AssistiveEnv(TaskInterface):
    """
        Interface with MujocoEnv
    """
    def __init__(self, args, env, set_params):
        super(AssistiveEnv).__init__()
        self.args = args
        self._env = env
        self.mujoco_model = Mujoco_model(set_params["controller_options"], **set_params["mujoco_options"])
        self._env_name = args.env
        
        if self.args.video_record:
            fps = int(set_params["mujoco_options"]["frame_skip"])
            self.video_writer = imageio.get_writer("{}.mp4".format(self.args.video_path), fps=fps)
            self._env.setup_camera(camera_width=1920//2, camera_height=1080//2)
        
        self.total_timesteps = 0
        self.episode_timesteps = 0
        self.episode_number = 0
        self.episode_reward = 0
        self.reward_scale = 0.01
         
        self.render = set_params["env_options"]["render"]
        
        self.max_episode_steps = set_params["alg_options"]["max_episode_steps"]
        self._state_dim = set_params["alg_options"]['context_dim']
        self._context_dim = set_params["alg_options"]['context_dim']
        self._latent_parameter_dim = set_params["alg_options"]['parameter_dim']
        self.action_dim = set_params["alg_options"]['action_dim']

        self.num_way_points = set_params["controller_options"]["num_waypoints"]
        
        self._latent_parameter_high = set_params["controller_options"]['stiffness_high']
        self._latent_parameter_low = set_params["controller_options"]['stiffness_low']
        
        self.tool_delta_pos = np.array(set_params["env_options"]["tool_delta_pos"][self._env_name])
        self.tool_delta_ori = np.array(set_params["env_options"]["tool_delta_ori"][self._env_name])

        self.delta_pose_high = self.mujoco_model.delta_pose_high
        self.delta_pose_low = self.mujoco_model.delta_pose_low
        self.delta_pose_scale = self.mujoco_model.delta_pose_scale

        self.delta_pose_params = np.zeros(self.action_dim)
        
        self.context = None
        self.target_pos = None
        self.target_ori = None
        self.target_euler = None

        self.target_pos_list = np.zeros((self.mujoco_model.num_way_points, 3))
        self.target_euler_list = np.zeros((self.mujoco_model.num_way_points, 3))

    def reset(self):
        """
            get initial context
        """
        print("Environment Reset !!!")
        print("+" * 50)
        if self.render:
            self._env.render()
    
        observation = self._env.reset()
        
        # reset joints and way points
        self.mujoco_ee_pose = self.mujoco_model.reset(observation['robot_joint_angles'])
        
        self.get_ee_state()
        
        self.target_pos_list[0] = self.mujoco_target_pos
        self.target_pos_list[1] = self.mujoco_target_pos_1

        self.target_euler_list[0] = self.mujoco_target_euler
        self.target_euler_list[1] = self.mujoco_target_euler_1

        self.set_waypoints(way_points_list=1,
                           target_pos=self.target_pos_list,
                           target_euler=self.target_euler_list)
        
        self.read_context()
        return observation
    
    def get_ee_state(self):
        self.tool_pos, self.tool_orient = self._env.get_tool_pose()
        self.tool_euler = transforms3d.euler.quat2euler(self.tool_orient, 'sxyz')
        print("tool_pos :", self.tool_pos, "tool_euler :", np.array(self.tool_euler),
              np.array(p.getEulerFromQuaternion(np.array(self.tool_orient), physicsClientId=self._env.id)))
        
        self.ee_pos, self.ee_ori = self._env.robot.get_ee_pose()
        self.ee_euler = transforms3d.euler.quat2euler(self.ee_ori, 'sxyz')
        print("ee_pos :", self.ee_pos, "ee_euler :", np.array(self.ee_euler),
              np.array(p.getEulerFromQuaternion(np.array(self.ee_ori), physicsClientId=self._env.id)))
        
        self.mujoco_model_ee = self.mujoco_model.get_ee_pose()
        print("mujoco_model_ee :", np.array(self.mujoco_model_ee))
 
        self.target_pos, self.target_ori, \
        self.target_pos_ref, self.target_orient_ref, \
        self.target_pos_tool, self.target_orient_tool = self._env.get_context()
        
        self.target_euler_ref = transforms3d.euler.quat2euler(self.target_orient_ref, 'sxyz')
        self.target_euler = transforms3d.euler.quat2euler(self.target_ori, 'sxyz')
        print("target_pos :", self.target_pos, "target_euler :", np.array(self.target_euler))
        print("reference_target_pos :", self.target_pos_ref,
              "reference_target_euler :", self.target_euler_ref)
        
        # self.delta_pos = self.target_pos_ref + self.tool_delta_pos - self.tool_pos
        # self.target_pos_ref_new, self.target_orient_ref_new = p.multiplyTransforms(
        #     self.target_pos_ref, self.target_orient_ref,
        #     self.ee_pos - self.tool_pos, [0, 0, 0, 1], physicsClientId=self._env.id)
        
        # self.tool_delta_pos, desired_ori = p.multiplyTransforms(
        #     self.ee_pos, self.ee_ori,
        #     self.ee_pos - self.tool_pos, [0, 0, 0, 1], physicsClientId=self._env.id)
        
        # + self.tool_delta_pos
        # print("tool_delta_pos :", self.tool_delta_pos, np.array(p.getEulerFromQuaternion(np.array(desired_ori), physicsClientId=self._env.id)))
        
        # self.initial_tool_delta_ori = np.array([-self.target_euler_ref[0], -self.target_euler_ref[1], self.target_euler_ref[2] - 1.57])
        # self.initial_tool_delta_ori = np.array([self.target_euler_ref[1], self.target_euler_ref[0], 0.0])
        
        self.pybullet_ori_euler = np.array(p.getEulerFromQuaternion(np.array(self.target_ori),
                                                                    physicsClientId=self._env.id))
        
        # self.delta_pos = self.target_pos_tool - self.ee_pos + self.tool_delta_pos
        # self.mujoco_target_pos = self.mujoco_ee_pose[:3] + self.delta_pos
        # self.mujoco_target_euler = self._desired_ori(self.mujoco_ee_pose[3:],
        #                                              self.pybullet_ori_euler + self.tool_delta_ori)
        
        # self.delta_pos_1 = self.target_pos_ref - self.ee_pos + self.tool_delta_pos
        # self.mujoco_target_pos_1 = self.mujoco_ee_pose[:3] + self.delta_pos_1
        # self.mujoco_target_euler_1 = self._desired_ori(self.mujoco_ee_pose[3:],
        #                                                np.array([0.0, 0.0, 0.0]))
        
        self.tool_delta_pos += self.delta_pose_params[:3]
        self.tool_delta_ori += self.delta_pose_params[3:]
        self.delta_pos = self.target_pos_tool - self.tool_pos + self.tool_delta_pos
        self.mujoco_target_pos = self.mujoco_ee_pose[:3] + self.delta_pos
        self.mujoco_target_euler = self._desired_ori(self.mujoco_ee_pose[3:],
                                                     self.pybullet_ori_euler + self.tool_delta_ori)

        self.delta_pos_1 = self.target_pos_ref - self.tool_pos + self.tool_delta_pos
        self.mujoco_target_pos_1 = self.mujoco_ee_pose[:3] + self.delta_pos_1
        self.mujoco_target_euler_1 = self._desired_ori(self.mujoco_ee_pose[3:],
                                                       np.array([0.0, 0.0, 0.0]))
    
    def read_context(self):
        """
            return context and target
        """
        self.target_pos, self.target_ori, _, _, _, _ = self._env.get_context()
        self.target_euler = transforms3d.euler.quat2euler(self.target_ori, 'sxyz')
        # self.context = np.concatenate((self.target_pos, self.target_ori), axis=0)
        self.context = np.concatenate((self.delta_pos, self.target_ori), axis=0)
        return self.context
    
    def send_movement(self, params):
        """
            send parameters
        """
        self.send_params(params)
        
        reward, context, info = self.run_single_trajectory(n=1)
        
        self.get_ee_state()
        
        return reward, context, info
    
    def set_waypoints(self, way_points_list=None, target_pos=None, target_euler=None):
        """
            set way points
        """
        # print("Mujoco_target_pos :", target_pos, 'target_euler :', target_euler)
        if way_points_list is None:
            self.target_pos_list = np.tile(target_pos, (self.mujoco_model.num_way_points, 1))
            self.target_euler_list = np.tile(target_euler, (self.mujoco_model.num_way_points, 1))

        self.mujoco_model.set_waypoints(self.target_pos_list, self.target_euler_list)
    
    def send_params(self, params):
        """
            set relations
        """
        self.mujoco_model.set_impedance_params(params[:self.num_way_points * self.action_dim + self.num_way_points - 1])
        self.delta_pose_params = params[-self.action_dim:]
        # self.delta_pose_params = params[-self.action_dim:] * self.delta_pose_scale[-self.action_dim:] \
        #                          + self.delta_pose_low[-self.action_dim:]
        # print("delta_pose_params :", self.delta_pose_params)
    
    def run_single_trajectory(self, n=None):
        average_reward = 0.0
        info = None
        context = None
        self.done = False
        self.episode_number += 1
        self.episode_timesteps = 0
        self.episode_reward = 0.0
        for i in range(n):
            obs = self.reset()
            context = self.read_context()
            joint_last = obs['robot_joint_angles'].copy()
            while True:
                robot_action = np.zeros(6)
                joint = self.mujoco_model.step(robot_action)
                
                # control human or not
                human_action = np.zeros(self._env.action_human_len)
                
                action = {'robot': joint.copy() - joint_last, 'human': human_action}
                # env.action_space_human.sample()
                # joint_list.append(joint[0].copy())
                # print("action :", action)
                np.putmask(action['robot'], action['robot'] > 3.14, 0.0)
                np.putmask(action['robot'], action['robot'] < -3.14, 0.0)
                
                obs, reward, self.done, info = self._env.step(action)
                # obs, reward, self.done, info = self._env.step(action, joint)
                
                # print("done robot :", self.done['robot'])
                # print("done human :", self.done['human'])
                joint_last = obs['robot_joint_angles'].copy()
                # print("joint last :", joint_last)
                
                # done_bool = 0 if self.episode_timesteps - 1 == self.max_episode_steps else float(self.done['robot'])
                if self.args.video_record:
                    img, _ = self._env.get_camera_image_depth()
                    self.video_writer.append_data(img)
                
                if self.render:
                    self._env.render()
                
                self.episode_reward += reward['robot']
                # self.episode_reward += reward['human']
                
                self.episode_timesteps += 1
                self.total_timesteps += 1
                
                if self.mujoco_model.get_state_dist() < 0.006:
                    break
                # print("mujoco_dist :", self.mujoco_model.get_state_dist())
                
                if self.done['__all__'] or info['robot']['task_success']:
                    break
             
            print("episode_reward :", self.episode_reward)
            
            # if self.done or self.episode_timesteps == self.max_episode_steps - 1:
            #     average_reward += self.episode_reward
            # else:
            #     average_reward += -300.0

            average_reward += self.episode_reward
            if info['robot']['task_success']:
                average_reward += 100
            else:
                average_reward += -100.0

            # print('joint :', joint.copy())
            # print('final :', obs['robot_joint_angles'].copy())
            print("Success :", info['robot']['task_success'], "Episode timesteps :", self.episode_timesteps,
                  "Reward :", np.around(average_reward / n, 4) * self.reward_scale)

            self.episode_number += 1
            self.episode_timesteps = 0
            self.episode_reward = 0.0
        
        return np.around(average_reward / n, 4) * self.reward_scale, context, info['robot']
        
    def get_demonstrations(self, num_traj=50):
        """
            generate demonstration samples
        """
        params_list = np.random.uniform(0, 1, size=(num_traj, self._latent_parameter_dim)) \
                      * (np.array(self._latent_parameter_high) - np.array(self._latent_parameter_low)) \
                      + np.array(self._latent_parameter_low)
        
        # # params_list = np.random.uniform(0, 1, size=(num_traj, self._env.latent_parameter_dim))
        print("params_list :", params_list)
        
        reward_list = []
        context_list = []
        success_list = []
        for i in range(num_traj):
            # context, _, _, _ = self._env.get_context()
            params = params_list[i, :]
            reward, context, info = self.send_movement(params)
            context_list.append(context.tolist())
            reward_list.append(np.copy(reward))
            success_list.append(info['task_success'])
        return np.hstack((np.array(context_list), params_list)), reward_list, success_list
        # return np.hstack((np.array(context_list), params_list)), reward_list
        # return [np.concatenate((np.array(context_list), params_list), axis=0)]

    def get_context_dim(self):
        return self._context_dim

    def get_impedance_dim(self):
        return self._latent_parameter_dim

    def _desired_ori(self, current_euler, rot_euler):
        # convert axis-angle value to rotation matrix
        # quat_error = trans.axisangle2quat(rot_euler)
        # rotation_mat_error = trans.quat2mat(quat_error)
        # curr_mat = trans.euler2mat(current_euler)
        rotation_mat_error = transforms3d.euler.euler2mat(rot_euler[0], rot_euler[1], rot_euler[2], 'sxyz')
        # rotation_mat_error = trans.quat2mat(quat_error)
        # curr_mat = trans.euler2mat(current_euler)
        curr_mat = transforms3d.euler.euler2mat(current_euler[0], current_euler[1], current_euler[2], 'sxyz')
        goal_orientation = np.dot(rotation_mat_error, curr_mat)
        # return trans.mat2euler(goal_orientation, 'sxyz')
        return transforms3d.euler.mat2euler(goal_orientation, 'sxyz')


class Mujoco_model():
    def __init__(self, controller_options, render=True, frame_skip=10, ratio=2.0,
                 stiff_scale=10, params_fixed=True):
        model_path = os.path.join(controller_options['model_root'], controller_options['model_path'])
        model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(model)
        
        self.controller = iMOGVIC(self.sim, **controller_options)
        
        self.action_dim = self.controller.action_dim
        
        self.render = render
        if self.render:
            self.viewer = mujoco_py.MjViewer(self.sim)
        
        self.num_way_points = controller_options['num_waypoints']
        self.stiffness_initial = controller_options['stiffness_initial']
        self.weight_initial = controller_options['weight_initial']
        self.frame_skip = frame_skip
        self.ratio = ratio
        self.stiffness_scale = stiff_scale
        self.params_fixed = params_fixed
        
        self.delta_pose_high = controller_options['stiffness_high']
        self.delta_pose_low = controller_options['stiffness_low']
        self.delta_pose_scale = np.array(self.delta_pose_high) - np.array(self.delta_pose_low)
        
        # way points
        self.pos_set_list = np.zeros((self.num_way_points, 3))
        self.euler_set_list = np.zeros((self.num_way_points, 3))
    
    def reset(self, initial_angles):
        """
            reset robot joints
        """
        # self.controller.update_initial_joints(initial_angles)
        self.sim_state = self.sim.get_state()
        self.sim_state.qpos[:7] = initial_angles
        qvel = np.zeros(7)
        self.sim_state.qvel[:7] = qvel
        self.sim.set_state(self.sim_state)
        self.sim.forward()
        
        self.controller.update_state()
        
        if self.render:
            self.viewer.render()
            render_frame(self.viewer, self.controller.ee_pose[:3], self.controller.ee_pose[3:])
        return self.controller.ee_pose
    
    def step(self, action):
        dt = self.sim.model.opt.timestep
        for _ in range(self.frame_skip):
            # update controller via imogic
            torque, V, pose_err, vel_err, stiffness_eqv, damping_eqv = self.controller.update_vic_torque()
            self.sim.data.ctrl[:7] = np.clip(torque[:7], -100, 100)
            self.sim.step()
            self.sim.forward()
            
            # render_frame(viewer, pos_set_list[0, :], quat_set_list[0, :])
            if self.render:
                self.viewer.render()
                render_frame(self.viewer, self.pos_set_list[0, :], self.euler_set_list[0, :])
        
        return self.controller.get_robot_joints()[0]
    
    def set_waypoints(self, target_pos_list, target_euler_list):
        """
            target_pose:
        """
        self.pos_set_list = target_pos_list
        self.euler_set_list = target_euler_list
        
        way_points_list = np.concatenate((self.pos_set_list, self.euler_set_list), axis=1)
        self.controller.set_way_points(way_points_list)
    
    def set_impedance_params(self, params):
        """
            stiffness, damping and weight
        """
        if self.params_fixed:
            print("params_fixed :")
            stiffness_params = self.stiffness_initial
            weight = self.weight_initial
            params = np.concatenate((stiffness_params, weight), axis=0)
            stiffness_list, damping_list, weight_list = self.send_params(params)
        else:
            stiffness_list, damping_list, weight_list = self.send_params(params)
        
        self.controller.set_params_direct(stiffness_list, damping_list, weight_list)
    
    def send_params(self, params):
        """
            set relations
        """
        params = np.clip(params, 0, 1)
        stiffness_list = params[:self.num_way_points * self.action_dim].reshape(
            self.num_way_points, -1) * self.stiffness_scale
        print("stiffness_list :", stiffness_list)
        damping_list = self.ratio * np.sqrt(stiffness_list)
        weight_list = np.ones(self.num_way_points)
        weight_list[1:] = params[self.num_way_points * self.action_dim:]
        return stiffness_list, damping_list, weight_list
    
    def get_joint_pos(self):
        return self.controller.get_robot_joints()[0]
    
    def get_ee_pose(self):
        return self.controller.ee_pose
    
    def get_state_dist(self):
        return np.linalg.norm(self.controller.state, ord=2)


class AssistiveDRL(object):
    def __init__(self, env, set_params, logdir):
        super(AssistiveDRL).__init__()
        
        self.logdir = logdir
        self.writer = FileWriter(logdir)
        
        self.render = set_params["render"]
        
        self._env = env
        
        self.mujoco_model = Mujoco_RL_model(set_params["controller_options"],
                                            **set_params["mujoco_options"])
        
        self.keys = set_params["controller_options"]["obs_keys"]
        
        self.state_dim = self._env.obs_robot_len + 7
        print("state_dim :", self.state_dim)
        
        self.observation_space = spaces.Box(low=np.array([-1000.0] * self.state_dim, dtype=np.float32),
                                       high=np.array([1000.0] * self.state_dim, dtype=np.float32),
                                       dtype=np.float32)
        
        self.action_dim = set_params["controller_options"]["action_dim"]
        self.action_space = spaces.Box(low=np.array([-1.0] * self.action_dim, dtype=np.float32),
                                       high=np.array([1.0] * self.action_dim, dtype=np.float32),
                                       dtype=np.float32)
        
        self.env_name = set_params["env"]
        self.max_episode_steps = set_params["env_options"]['max_episode_steps'][self.env_name]
        self.metadata = None
        self.reward_range = (-float(100), float(100))
    
        self.total_steps = 0
        self.episode_reward = 0.0

    def reset(self):
        """
            get initial context
        """
        if self.render:
            self._env.render()
         
        observation = self._env.reset()
        # print('Observation :', observation['robot'].shape)
        # print('robot_joint_angles :', observation['robot_joint_angles'])
        self.last_joint = observation['robot_joint_angles']

        # set way points : target pose
        start_pos, start_ori = self._env.robot.get_ee_pose()
        start_euler = transforms3d.euler.quat2euler(start_ori, 'sxyz')
        # print("Pybullet start_pos :", start_pos, "start_euler :", start_euler)

        # reset joints
        mujoco_ee_pose = self.mujoco_model.reset(observation['robot_joint_angles'])
        # print("Mujoco_ee_pos :", mujoco_ee_pose)
        
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        return self._flatten_obs(observation)

    def step(self, action):
        self.get_ee_state()
        
        target_ee_ori_mat = self._desired_ori(current_euler=self.ee_pose[3:], rot_euler=action[3:])
        joint = self.mujoco_model.step(action, set_ori=target_ee_ori_mat)
        
        human_action = np.zeros(self._env.action_human_len)

        action = {'robot': joint.copy() - self.last_joint, 'human': human_action}  # env.action_space_human.sample()
        
        # print("action :::", action)
        np.putmask(action['robot'], action['robot'] > 3.14, 0.0)
        np.putmask(action['robot'], action['robot'] < -3.14, 0.0)
        observation, reward, done, info = self._env.step(action)
        
        if self.render:
            self._env.render()
        
        self.last_joint = observation['robot_joint_angles'].copy()
        
        reward, done, info = self._flatten_reward(reward, done, info)
            
        self.episode_steps += 1
        self.total_steps += 1
        
        self.episode_reward += reward
        
        # self.episode_steps > self.max_episode_steps
        if done or info['task_success']:
            done = True
            if info['task_success']:
                reward += 10
                self.episode_reward += reward
            else:
                reward += -10
                self.episode_reward += reward

            if done:
                self.writer.add_scalar('train_episode_reward', self.episode_reward, self.total_steps)
                self.writer.add_scalar('success_rate', info['task_success'], self.total_steps)
                self.episode_reward = 0
    
                # Clear the episode_info dictionary
                self.episode_info = dict()
        
        return self._flatten_obs(observation), reward, done, info
    
    def get_ee_state(self):
        self.ee_pose = self.mujoco_model.get_ee_pose()
        self.ee_pos = self.ee_pose[:3]
        self.ee_ori_mat = transforms3d.euler.euler2mat(self.ee_pose[3], self.ee_pose[4], self.ee_pose[5])
    
    def view_render(self):
        self._env.render()

    def _flatten_obs(self, obs_dict):
        """
            Filters keys of interest out and concatenate the information.
        """
        ob_lst = []
        ob_lst.append(obs_dict["robot"])
        ob_lst.append(np.cos(obs_dict["robot_joint_angles"]))
        ob_lst.append(np.sin(obs_dict["robot_joint_angles"]))
        # print(np.sin(obs_dict["robot_joint_angles"]))
        # for key in self.keys:
        #     if key in obs_dict:
        #         ob_lst.append(np.array(obs_dict[key]).flatten())
        
        return np.concatenate(ob_lst)

    def _flatten_reward(self, reward, done, info):
        # {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}
    
        return reward['robot'], done['__all__'], info['robot']
        
    def _desired_ori(self, current_euler, rot_euler):
        # convert axis-angle value to rotation matrix
        # quat_error = trans.axisangle2quat(rot_euler)
        # rotation_mat_error = trans.quat2mat(quat_error)
        # curr_mat = trans.euler2mat(current_euler)
        rotation_mat_error = transforms3d.euler.euler2mat(rot_euler[0], rot_euler[1], rot_euler[2], 'sxyz')
        # rotation_mat_error = trans.quat2mat(quat_error)
        # curr_mat = trans.euler2mat(current_euler)
        curr_mat = transforms3d.euler.euler2mat(current_euler[0], current_euler[1], current_euler[2], 'sxyz')
        goal_orientation = np.dot(rotation_mat_error, curr_mat)
        # return trans.mat2euler(goal_orientation, 'sxyz')
        # return transforms3d.euler.mat2euler(goal_orientation, 'sxyz')
        return goal_orientation


class Mujoco_RL_model():
    def __init__(self,
                 controller_options,
                 render=True,
                 frame_skip=10,
                 ratio=2.0,
                 stiff_scale=10,
                 params_fixed=True):
        
        model_path = os.path.join(controller_options['model_root'], controller_options['model_path'])
        model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(model)
        
        self.controller_name = controller_options['controller_name']
        controller_path = os.path.join('/home/zhimin/code/5_thu/rl-robotic-assembly-control/envs/robosuite/robosuite/',
                                       'controllers/config/{}.json'.format(self.controller_name.lower()))
        
        self.controller_config = load_controller_config(custom_fpath=controller_path)
        self.controller_config['sim'] = self.sim
        self.controller_config["eef_name"] = "ee_site"
        self.controller_config['robot_joints'] = controller_options["controlled_joints"]
        self._get_index(self.controller_config['robot_joints'])
        
        self.controller_config["joint_indexes"] = {
            "joints": self._ref_joint_indexes,
            "qpos": self._ref_joint_pos_indexes,
            "qvel": self._ref_joint_vel_indexes
        }
        
        self.controller_config["impedance_mode"] = controller_options["impedance_mode"]
        self.controller_config["kp_limits"] = controller_options["kp_limits"]
        self.controller_config["damping_limits"] = [0, 10]
        self.controller_config["actuator_range"] = self._torque_limits()
        self.controller_config["policy_freq"] = 20
        
        self.controller = controller_factory(self.controller_name, self.controller_config)
        
        self.action_dim = controller_options['action_dim']
        self.frame_skip = frame_skip
        
        self.render = render
        if self.render:
            self.viewer = mujoco_py.MjViewer(self.sim)
        
        # self.num_way_points = controller_options['num_waypoints']
        # self.stiffness_initial = controller_options['stiffness_initial']
        # self.weight_initial = controller_options['weight_initial']
        # self.ratio = ratio
        # self.stiffness_scale = stiff_scale
        # self.params_fixed = params_fixed
    
    def reset(self, initial_angles):
        """
            reset robot joints
        """
        # self.controller.update_initial_joints(initial_angles)
        self.sim_state = self.sim.get_state()
        self.sim_state.qpos[:7] = initial_angles
        qvel = np.zeros(7)
        self.sim_state.qvel[:7] = qvel
        self.sim.set_state(self.sim_state)
        self.sim.forward()

        self.controller.update(force=True)
        self.controller.reset_goal()
        
        if self.render:
            self.viewer.render()
            
        self.episode_step = 0
        
        return self.get_ee_pose()
    
    def step(self, action, set_pos=None, set_ori=None):
        dt = self.sim.model.opt.timestep
        self._set_goal(action, set_pos=set_pos, set_ori=set_ori)
        
        for _ in range(self.frame_skip):
            # print("set_pos", self.controller.goal_pos, "set_ori", self.controller.goal_ori)
            torques = self._control(action, set_pos=set_pos, set_ori=set_ori)
            # print("torques :", torques)
            self.sim.data.ctrl[:7] = torques[:7]
            self.sim.step()
            self.sim.forward()
            
            # render_frame(viewer, pos_set_list[0, :], quat_set_list[0, :])
            if self.render:
                self.viewer.render()
        
        # print("dist :", np.linalg.norm(self.controller.state, ord=2))
        return self.get_joint_state()
    
    def _set_goal(self, action, set_pos=None, set_ori=None):
        # clip actions into valid range
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))
    
        arm_action = action
    
        # Update the controller goal if this is a new policy step
        self.controller.set_goal(arm_action, set_pos=set_pos, set_ori=set_ori)
        
    def _control(self, action, set_pos=None, set_ori=None):
        # # clip actions into valid range
        # assert len(action) == self.action_dim, \
        #     "environment got invalid action dimension -- expected {}, got {}".format(
        #         self.action_dim, len(action))
        #
        # arm_action = action
        #
        # # Update the controller goal if this is a new policy step
        # self.controller.set_goal(arm_action, set_pos=set_pos, set_ori=set_ori)
        
        # Now run the controller for a step
        torques = self.controller.run_controller()
        # print("torques :", torques)
        
        # Clip the torques
        low, high = self._torque_limits()
        self.torques = np.clip(torques, low, high)
    
        # # Apply joint torque control
        # self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques
        return self.torques
    
    def get_joint_state(self):
        # , self.controller.joint_vel
        self.controller.update()
        return self.controller.joint_pos
    
    def get_ee_pose(self):
        self.ee_pos = self.controller.ee_pos
        self.ee_ori_euler = transforms3d.euler.mat2euler(self.controller.ee_ori_mat, 'sxyz')
        return np.hstack([self.ee_pos, self.ee_ori_euler])
    
    def _get_index(self, robot_joints):
        # indices for joint indexes
        self._ref_joint_indexes = [
            self.sim.model.joint_name2id(joint) for joint in robot_joints
        ]
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in robot_joints
        ]
        self._ref_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in robot_joints
        ]
    
    def _torque_limits(self):
        """
        Torque lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) torque values
                - (np.array) maximum (high) torque values
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 1]
        return low, high
    
    def viewer_render(self):
        self.viewer.render()


class MujocoEnv(TaskInterface):
    """
        Interface with MujocoEnv
    """
    
    def __init__(self, env, params):
        super(MujocoEnv).__init__()
        
        self._env = env
        
        self.total_timesteps = 0
        self.episode_timesteps = 0
        self.episode_number = 0
        self.episode_reward = 0
        self.reward_scale = params['alg_options']['reward_scale']
        self.stiffness_scale = params['alg_options']['stiffness_scale']
        self.ratio = params['alg_options']['ratio']
        
        self.max_episode_steps = self._env._max_episode_steps
        
        self._state_dim = self._env.context_dim
        self._context = None
        self.num_way_points = self._env.num_waypoints
        self.action_dim = self._env.action_dim
        
        self.target_pos = None
        self.target_quat = None
        self.target_euler = None
    
    def get_context_dim(self):
        return self._env.context_dim
    
    def get_impedance_dim(self):
        return self._env.parameter_dim
    
    def read_context(self):
        # context, target_pos, target_quat, target_euler = self._env.get_context()
        # return context, target_pos, target_quat, target_euler
        if self._context is not None:
            return self._context
        else:
            exit()
    
    def send_movement(self, params, render=False):
        """
            send parameters
        """
        stiffness_list, damping_list, weight_list = self.send_params(params)
        # stiffness_list = stiffness_list.reshape(self.num_way_points, self.action_dim)
        # damping_list = damping_list.reshape(self.num_way_points, self.action_dim)
        # print("stiffness_list :", stiffness_list)
        # print("damping_list :", damping_list)
        # print("weight_list :", weight_list)
        self._env.set_params(stiffness_list, damping_list, weight_list)
        reward, context, info = self.run_single_trajectory(n=1, render=render)
        return reward, context, info
    
    def set_waypoints(self, way_points_list=None):
        if way_points_list is None:
            pos_set_list = np.zeros((self.num_way_points, 3))
            quat_set_list = np.zeros((self.num_way_points, 3))
            for i in range(self.num_way_points):
                pos_set_list[i, :] = self.target_pos
                quat_set_list[i, :] = self.target_euler
            
            way_points_list = np.concatenate((pos_set_list, quat_set_list), axis=1)
            self._env.set_waypoints(way_points_list)
        else:
            print("Please give the waypoints !!!")
    
    def send_params(self, params):
        """
            set relations
        """
        params = np.clip(params, 0, 1)
        # print("params :", params)
        stiffness_list = params[:self.num_way_points * self.action_dim].reshape(self.num_way_points,
                                                                                -1) * self.stiffness_scale
        damping_list = self.ratio * np.sqrt(stiffness_list)
        weight_list = params[self.num_way_points * self.action_dim:]
        return stiffness_list, damping_list, weight_list
    
    def run_single_trajectory(self, n=None, render=False):
        average_reward = 0.0
        info = None
        for i in range(n):
            self.reset()
            
            if render:
                self._env.render()
                time.sleep(1.0)
            
            # set way points
            self.set_waypoints(way_points_list=None)
            while self.episode_timesteps < self.max_episode_steps:
                action = np.zeros(6)
                new_obs, reward, self.done, info = self._env.step_imogic(action)
                
                done_bool = 0 if self.episode_timesteps - 1 == self._env._max_episode_steps else float(self.done)
                
                if render:
                    self._env.render()
                
                self.episode_reward += reward
                self.episode_timesteps += 1
                self.total_timesteps += 1
                
                if self.done:
                    break
            
            if done_bool:
                average_reward += self.episode_reward
            else:
                average_reward += -100.0
            
            print("Done :", done_bool, "Episode timesteps :", self.episode_timesteps, "Reward :",
                  np.around(average_reward / n, 4) * self.reward_scale)
            self.episode_number += 1
            self.episode_timesteps = 0
            self.episode_reward = 0.0
            done_bool = 0
        
        return np.around(average_reward / n, 4) * self.reward_scale, self._context, info
    
    def reset(self):
        """
            get context after env reset
        """
        self._env.reset()
        self._context, self.target_pos, self.target_quat, self.target_euler = \
            self._env.get_context()
    
    def get_demonstrations(self, num_traj=50, render=False):
        """
            generate demonstration samples
        """
        params_list = np.random.uniform(0, 1, size=(num_traj, self._env.latent_parameter_dim)) \
                      * (np.array(self._env.latent_parameter_high) - np.array(self._env.latent_parameter_low)) \
                      + np.array(self._env.latent_parameter_low)
        
        # # params_list = np.random.uniform(0, 1, size=(num_traj, self._env.latent_parameter_dim))
        # print("params_list :", params_list.shape)
        
        reward_list = []
        context_list = []
        for i in range(num_traj):
            params = params_list[i, :]
            reward, context, info = self.send_movement(params, render=render)
            # context, _, _, _ = self._env.get_context()
            context_list.append(context.tolist())
            # print("info :", info)
            reward_list.append(np.copy(reward))
        
        return np.hstack((np.array(context_list), params_list))
    # return np.hstack((np.array(context_list), params_list)), reward_list
    # return [np.concatenate((np.array(context_list), params_list), axis=0)]


class PybulletEnv(TaskInterface):
    
    def __init__(self, env, state_dim, action_dim, n_features):
        super(PybulletEnv).__init__()
        
        self._group = Group("pybullet", ["joint%d" % i for i in range(action_dim)])
        self._space = ClassicSpace(self._group, n_features)

        self._env = env
        self.task = self._env.task
        
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._n_features = n_features
        self._headless = True
        self._env.latent_parameter_dim = self._n_features * self._action_dim
        
        # obs_config = ObservationConfig()
        # obs_config.set_all_low_dim(True)
        # obs_config.set_all_high_dim(False)
        # self._obs_config = obs_config
        #
        # action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
        # self._action_mode = action_mode
        
        self._obs = None
        self.render = True
    
    def get_context_dim(self):
        return self._state_dim
    
    def get_impedance_dim(self):
        return self._env.latent_parameter_dim
    
    def read_context(self):
        # self._env.reset()
        tool_pos, tool_orient = self._env.get_tool_pose()
        target_pos, target_orient, _, _, _, _ = self._env.get_context()
        delta_pos = target_pos - tool_pos
        return np.concatenate((delta_pos, target_orient), axis=0)
    
    def get_demonstrations(self, num_traj=50):
        # file = "parameters/%s_%d.npy" % (self.env_name, self._space.n_features)
        # try:
        #     return np.load(file)
        # except:
        #     raise Exception("File %s not found. Please consider running 'dataset_generator.py'" % file)
        
        # generate random parameters and collect data
        # params_list = np.random.uniform(0, 1, size=(num_traj, self._env.latent_parameter_dim)) \
        #               * (np.array(self._env.latent_parameter_high) - np.array(self._env.latent_parameter_low)) \
        #               + np.array(self._env.latent_parameter_low)
        
        params_list = np.random.uniform(0, 1, size=(num_traj, self._n_features * self._action_dim))
        print("params_list :", params_list.shape)
        
        reward_list = []
        context_list = []
        duration = 5
        for i in range(num_traj):
            print('+' * 25, i)
            self.reset()
            context = self.read_context()
            context_list.append(context.tolist())
            params = params_list[i, :]
            reward, info = self.send_movement(params, duration)
            reward_list.append(np.copy(reward))
        return np.hstack((np.array(context_list), params_list))
    
    def send_movement(self, weights, duration):
        mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
        duration = 1 if duration < 0 else duration
        
        if self._headless:
            trajectory = mp.get_full_trajectory(duration=min(duration, 1), frequency=200)
        else:
            trajectory = mp.get_full_trajectory(duration=5 * duration, frequency=200)
        
        # return trajectory
        
        tot_reward = 0.
        success = 0
        epi_reward = 0.
        info = dict()
        # self._env.render()
        # self._env.reset()
        for a1 in trajectory.values:  # a2 in zip(trajectory.values[:-1, :], trajectory.values[1:, :]):
            joint = a1  # (a2-a1) * 20.
            print("robot joint :", joint)
            action = {'robot': joint - self.joint, 'human': np.zeros(self._env.action_human_len)} # self._env.action_space_human.sample()
            obs, reward, terminate, info = self._env.step(action)
            self.joint = obs['robot_joint_angles'].copy()
            if self.render:
                self._env.render()
             
            if reward == 1. or terminate == 1.:
                if reward == 1.:
                    success = 1.
                break
        
        # tot_reward, success = self._stop(action, success)
        info['tot_reward'] = tot_reward
        info['epi_reward'] = epi_reward
        return success, info
    
    def reset(self):
        self._env.render()
        self._obs = self._env.reset()
        self.joint = self._obs['robot_joint_angles'].copy()

# def _stop(self, joint_gripper, previous_reward):
#     if previous_reward == 0.:
#         success = 0.
#         for _ in range(50):
#             obs, reward, terminate = self.task.step(joint_gripper)
#             if reward == 1.0:
#                 success = 1.
#                 break
#         return self.task._task.get_dense_reward(), success
#     return self.task._task.get_dense_reward(), 1.


class RLBenchBox(TaskInterface):

    def __init__(self, task_class, state_dim, n_features, headless=True):

        super().__init__(n_features)
        self._group = Group("rlbench", ["d%d" % i for i in range(7)] + ["gripper"])
        self._space = ClassicSpace(self._group, n_features)
        
        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        self._obs_config = obs_config

        self._state_dim = state_dim
        self._headless = headless

        action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
        self._task_class = task_class
        self._action_mode = action_mode
        self.env = Environment(action_mode, "", obs_config, headless=headless)
        self.env.launch()

        self.task = self.env.get_task(task_class)
        self._obs = None

    def get_context_dim(self):
        return self._state_dim

    def read_context(self):
        return self._obs[1].task_low_dim_state

    def get_demonstrations(self):
        file = "parameters/%s_%d.npy" % (self.task.get_name(), self._space.n_features)
        try:
            return np.load(file)
        except:
            raise Exception("File %s not found. Please consider running 'dataset_generator.py'" % file)
 
    def _stop(self, joint_gripper, previous_reward):
        if previous_reward == 0.:
            success = 0.
            for _ in range(50):
                obs, reward, terminate = self.task.step(joint_gripper)
                if reward == 1.0:
                    success = 1.
                    break
            return self.task._task.get_dense_reward(), success
        return self.task._task.get_dense_reward(), 1.

    def send_movement(self, weights, duration):
        mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
        duration = 1 if duration < 0 else duration
        if self._headless:
            trajectory = mp.get_full_trajectory(duration=min(duration, 1), frequency=200)
        else:
            trajectory = mp.get_full_trajectory(duration=5 * duration, frequency=200)
        
        tot_reward = 0.
        success = 0
        for a1 in trajectory.values:  # , a2 in zip(trajectory.values[:-1, :], trajectory.values[1:, :]):
            joint = a1  # (a2-a1)*20.
            joint_gripper = joint
            obs, reward, terminate = self.task.step(joint_gripper)
            if reward == 1. or terminate == 1.:
                if reward == 1.:
                    success = 1.
                break
        tot_reward, success = self._stop(joint_gripper, success)
        return success, tot_reward

    def reset(self):
        self._obs = self.task.reset()


class Forward2DKinematics:

    def __init__(self, d1, d2):
        self._d1 = d1
        self._d2 = d2

    def _link(self, d):
        return np.array([d, 0.])

    def _rot(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    def get_forward(self, theta1, theta2):
        x1 = self._rot(theta1) @ self._link(self._d1)
        r1 = self._rot(theta1) @ self._rot(0.)
        r2 = self._rot(theta2) @ r1
        x2 = r2 @ self._link(self._d2) + x1
        return x2

    def get_full_forward(self, theta1, theta2):
        x1 = self._rot(theta1) @ self._link(self._d1)
        r1 = self._rot(theta1) @ self._rot(0.)
        r2 = self._rot(theta2) @ r1
        x2 = r2 @ self._link(self._d2) + x1
        return x1, x2

    def get_loss(self, theta1, theta2, goal):
        ref = self.get_forward(theta1, theta2)
        return np.mean((ref - goal)**2)

    def jac(self, theta1, theta2, goal, delta=1E-5):
        ref = self.get_loss(theta1, theta2, goal)
        j1 = (self.get_loss(theta1 + delta, theta2, goal) - ref)/delta
        j2 = (self.get_loss(theta1, theta2+delta, goal) - ref)/delta
        return np.array([j1, j2])

    def get_trajectory(self, theta1, theta2, goal, v=0.1):
        conf = [np.array([theta1, theta2])]
        for _ in range(200):
            conf.append(conf[-1]-v*self.jac(conf[-1][0], conf[-1][1], goal))
        return conf, [self.get_forward(c[0], c[1]) for c in conf]


class Reacher2D(TaskInterface):

    def __init__(self,  n_features, points=0, headless=True):

        super().__init__(n_features)
        self._group = Group("reacher2d", ["j%d" % i for i in range(2)])
        self._space = ClassicSpace(self._group, n_features)

        self._state_dim = 2
        self._headless = headless

        self._n_points = points
        self._goals = [self._point(3/2, np.pi/8),
                       self._point(1., np.pi/2 + np.pi/8),
                       self._point(2/3, np.pi + np.pi/4),
                       self._point(1/2, 3/2*np.pi + np.pi/6)]
        self._kinematics = Forward2DKinematics(1., 1.)

        self._context = None

    def _point(self, d, theta):
        return d*np.array([np.cos(theta), np.sin(theta)])

    def _generate_context(self, goal=None):
        if self._n_points == 0:
            d = np.random.uniform(0, 1)
            a = np.random.uniform(-np.pi, np.pi)
            return self._point(d, a)
        else:
            if goal is None:
                k = np.random.choice(range(self._n_points))
            else:
                k = goal
            g = self._goals[k]
            d = np.random.uniform(0, 1/5)
            a = np.random.uniform(-np.pi, np.pi)
            return g + self._point(d, a)

    def give_example(self, goal=None):
        goal = self._generate_context(goal)
        conf, traj = self._kinematics.get_trajectory(0., 0., goal)
        return goal, conf, traj

    def _generate_demo(self):
        goal = self._generate_context()
        conf, traj = self._kinematics.get_trajectory(0., 0., goal)
        trajectory = NamedTrajectory(*self._group.refs)
        for c in conf:
            trajectory.notify(duration=1/100.,
                              j0=c[0], j1=c[1])
        return goal, np.array([3.]), LearnTrajectory(self._space, trajectory).get_block_params()

    def get_context_dim(self):
        return self._state_dim

    def read_context(self):
        return self._context

    def get_demonstrations(self):
        return np.array([np.concatenate(self._generate_demo(), axis=0) for _ in range(100)])

    def send_movement(self, weights, duration):
        mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
        duration = 1 if duration < 0 else duration
        trajectory = mp.get_full_trajectory(duration=duration, frequency=200)

        vals = trajectory.get_dict_values()

        reward = -self._kinematics.get_loss(vals["j0"][-1], vals["j1"][-1], self._context)

        return reward, reward

    def reset(self):
        self._context = self._generate_context()


class ObstacleRectangle:

    def __init__(self, x, y, dx, dy):
        self.x1 = x
        self.x2 = x + dx
        self.dx = dx
        self.y1 = y
        self.y2 = y + dy
        self.dy = dy
        self._patch = patches.Rectangle((x, y), dx, dy,
                                        linewidth=1, edgecolor='r', facecolor='r')

    def check_collision_point(self, point):
        if self.x1 <= point[0] <= self.x2:
            if self.y1 <= point[1] <= self.y2:
                return True
        return False

    def check_collision_points(self, points):
        for point in points:
            if self.check_collision_point(point):
                return True
        return False

    def draw(self, ax):
        self._patch = patches.Rectangle((self.x1, self.y1), self.dx, self.dy,
                                        linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(self._patch)


def _positive_range(angle):
    ret = angle
    while ret < 0.:
        ret += 2 * np.pi
    while ret > 2 * np.pi:
        ret -= 2 * np.pi
    return ret


def _2p_range(angle):
    ret = angle
    while ret < -np.pi:
        ret += 2 * np.pi
    while ret > np.pi:
        ret -= 2 * np.pi
    return ret


def get_angle_between(angle_1, angle_2):
    _angle_1 = _positive_range(angle_1)
    _angle_2 = _positive_range(angle_2)
    if np.abs(_angle_1 - _angle_2) < np.pi:
        return np.abs(_angle_1 - _angle_2)
    else:
        return 2*np.pi - np.abs(_angle_1 - _angle_2)


def get_mid_angle(angle_1, angle_2, length):
    _angle_1 = _positive_range(angle_1)
    _angle_2 = _positive_range(angle_2)
    if np.abs(_angle_1 - _angle_2) < np.pi:
        ret = _angle_1 + np.clip(_angle_2 - _angle_1, -length, length)
    else:
        if _angle_2 > _angle_1:
            delta = get_angle_between(_angle_2, _angle_1)
            delta = min(delta, length)
            ret = _angle_1 - delta
        else:
            delta = get_angle_between(_angle_2, _angle_1)
            delta = min(delta, length)
            ret = _angle_1 + delta
    ret = _2p_range(ret)
    return ret


def sampling():
    return np.random.uniform(-np.pi * np.ones(2), np.pi * np.ones(2))
# print(get_angle_between(-np.pi+0.1, np.pi-0.1))


class ObstacleReacher2d(TaskInterface):

    def __init__(self,  n_features, headless=True):

        super().__init__(n_features)
        self._group = Group("reacher2d", ["j%d" % i for i in range(2)])
        self._space = ClassicSpace(self._group, n_features)

        self._state_dim = 2
        self._headless = headless

        self._obstacle = ObstacleRectangle(0.5, 0.5, 0.25, 0.25)

        self._kinematics = Forward2DKinematics(1., 1.)

        self._rrt_star = RRTStar(np.array([0., 0.]), self.close_to_goal, sampling, 0.05,
                                 self.get_configuration_distance,
                                 collision_detector=self.check_collision,
                                 goal_distance=self.distance_to_goal,
                                 get_mid_point=self.get_mid_configuration,
                                 star=True,
                                 star_distance=0.1)

        self._context = None
        self.reset()

    def get_configuration_distance(self, conf_1, conf_2):
        d1 = get_angle_between(conf_1[0], conf_2[0])
        d2 = get_angle_between(conf_1[1], conf_2[1])
        return np.sqrt(d1**2 + d2**2)

    def get_mid_configuration(self, conf_1, conf_2, length=0.1):
        d1 = get_mid_angle(conf_1[0], conf_2[0], length)
        d2 = get_mid_angle(conf_1[1], conf_2[1], length)
        return np.array([d1, d2])

    def check_collision(self, configuration):
        x1, x2 = self._kinematics.get_full_forward(configuration[0], configuration[1])
        points_l1 = np.linspace(np.zeros_like(x1), x1)
        if self._obstacle.check_collision_points(points_l1):
            # print("Collision conf", configuration[0], configuration[1])
            # print("Collision points", x1, x2)
            return True
        points_l2 = np.linspace(x1, x2)
        if self._obstacle.check_collision_points(points_l2):
            # print("Collision conf", configuration[0], configuration[1])
            # print("Collision points", x1, x2)
            return True
        return False

    def distance_to_goal(self, configuration):
        x = self._kinematics.get_forward(configuration[0], configuration[1])
        return np.sqrt(np.sum((x - self._context)**2))

    def close_to_goal(self, configuration):
        x = self._kinematics.get_forward(configuration[0], configuration[1])
        return np.sqrt(np.sum((x - self._context)**2)) < 0.1

    def close_to_goal_env(self, configuration):
        # print("Goal was is", self._context)
        # print("Configuration is", configuration)
        # print("Reached position is", self._kinematics.get_forward(configuration[0], configuration[1]))
        x = self._kinematics.get_forward(configuration[0], configuration[1])
        return np.sqrt(np.sum((x - self._context)**2)) < 0.4

    def draw(self, configuration, ax, alpha=1.0):
        x1, x2 = self._kinematics.get_full_forward(configuration[0], configuration[1])
        ax.plot([0, x1[0]], [0, x1[1]], c='gray', alpha=alpha)
        ax.scatter(x1[0], x1[1], color='gray', alpha=alpha, s=2)
        ax.plot([x2[0], x1[0]], [x2[1], x1[1]], c='gray', alpha=alpha)
        ax.scatter(x2[0], x2[1], color='gray', alpha=alpha, s=2)

    def draw_goal(self, ax):
        circle = patches.Circle(self._context, 0.1, edgecolor='green', facecolor='white')
        ax.add_patch(circle)

    def _point(self, d, theta):
        return d*np.array([np.cos(theta), np.sin(theta)])

    def _generate_context(self, goal=None):
        d = np.random.uniform(0, 1.8)
        a = np.random.uniform(-np.pi, np.pi)
        ret = self._point(d, a)
        while self._obstacle.check_collision_point(ret):
            d = np.random.uniform(0, 1.8)
            a = np.random.uniform(-np.pi, np.pi)
            ret = self._point(d, a)
        return ret

    def give_example(self, goal=None):
        # TODO: change
        goal = self._generate_context(goal)
        conf, traj = self._kinematics.get_trajectory(0., 0., goal)
        return goal, conf, traj

    def _generate_demo(self, reuse_rrt_graph=True):
        self.reset()
        goal = self.read_context()

        graph = self._rrt_star.graph if reuse_rrt_graph else None

        self._rrt_star = RRTStar(np.array([0., 0.]), self.close_to_goal, sampling, 0.05,
                self.get_configuration_distance,
                collision_detector=self.check_collision,
                goal_distance=self.distance_to_goal,
                get_mid_point=self.get_mid_configuration,
                star=True,
                graph=graph,
                star_distance=0.1)

        if len(self._rrt_star.graph.all_nodes) == 0:
            for _ in range(5000):
                self._rrt_star.add_point()

        self._rrt_star.evaluate()

        if self._rrt_star.is_goal_reached():
            print("RRT SUCCESS")
        else:
            print("RRT FAIL")

        last_node = self._rrt_star.closest_node
        traj = []
        for node in last_node.get_path_to_origin():
            pos = node.position
            traj.append(pos)

        trajectory = NamedTrajectory(*self._group.refs)
        for c in traj:
            trajectory.notify(duration=1/100.,
                              j0=c[0], j1=c[1])
        return goal, np.array([len(traj)/100.]), trajectory

    def get_context_dim(self):
        return self._state_dim

    def read_context(self):
        return self._context

    def save_demonstration(self):
        demos = [] #np.array([np.concatenate(self._generate_demo(), axis=0) for _ in range(5)])
        for i in range(100000):
            # current_demos = np.array([np.concatenate(self._generate_demo(), axis=0) for _ in range(5)])
            start = time.time()
            if i % 20 == 0:
                goal, duration, trajectory = self._generate_demo(reuse_rrt_graph=False)
            else:
                goal, duration, trajectory = self._generate_demo(reuse_rrt_graph=True)
            params = LearnTrajectory(self._space, trajectory).get_block_params()
            current_demo = np.concatenate([goal, duration, params], axis=0)
            trajectory.save("trajectories/trajectory_%d.npy" % i)
            demos.append(current_demo)

            print("demo-time: %f" % (time.time() - start))
            if i % 5 == 0:
                np.save("obstacle.npy", demos)
        demo = np.load("core/demonstrations/reacher2d_obstacle.npy")
        print("Loaded demo", demo.shape)
        return demo

    def get_demonstrations(self):
        demo = np.load("core/demonstrations/reacher2d_obstacle.npy")
        print("Loaded demo", demo.shape)
        return demo

    def get_success_demo(self):
        demos = self.get_demonstrations()
        ret = []
        rew = []
        for i, demo in enumerate(demos):
            self._context = demo[:2]
            traj = LoadTrajectory("trajectories/trajectory_%d.npy" % i)
            traj_val = traj.get_dict_values()
            conf = np.array([traj_val["j0"][-1], traj_val["j1"][-1]])
            # for v0, v1 in zip(vals["j0"], vals["j1"]):
            #     if self.check_collision(np.array([v0, v1])):
            #         return False, -1.

            rew.append(-self._kinematics.get_loss(traj_val["j0"][-1], traj_val["j1"][-1], self._context))
            ret.append(self.close_to_goal(conf))

        return np.sum(ret)/len(ret), np.sum(rew)/len(rew)

    def send_movement(self, weights, duration):
        mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
        duration = 1 if duration < 0 else duration
        trajectory = mp.get_full_trajectory(duration=duration, frequency=50)

        vals = trajectory.get_dict_values()

        for v0, v1 in zip(vals["j0"], vals["j1"]):
            if self.check_collision(np.array([v0, v1])):
                return False, -1.

        reward = -self._kinematics.get_loss(vals["j0"][-1], vals["j1"][-1], self._context)

        return self.close_to_goal_env(np.array([vals["j0"][-1], vals["j1"][-1]])), reward

    def reset(self):
        self._context = self._generate_context()