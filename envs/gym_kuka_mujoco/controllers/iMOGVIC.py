import os
import numpy as np
from gym import spaces
import mujoco_py

from envs.gym_kuka_mujoco.envs.assets import kuka_asset_dir
from envs.gym_kuka_mujoco.utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat
from envs.mujoco.utils.kinematics import forwardKinSite, forwardKinJacobianSite, forwardVelKinSite
from .base_controller import BaseController
from . import register_controller
from envs.gym_kuka_mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, \
    get_joint_indices
from envs.mujoco.mujoco_config import MujocoConfig
from envs.gym_kuka_mujoco.utils.control_utils import *
from envs.gym_kuka_mujoco.utils.transform_utils import *

from collections.abc import Iterable
import transforms3d as transforms3d


class iMOGVIC(BaseController):
    def __init__(self,
                 sim,
                 pos_scale=1.0,
                 rot_scale=0.3,
                 pos_limit=1.0,
                 rot_limit=1.0,
                 model_root=None,
                 model_path='full_kuka_no_collision_no_gravity.xml',
                 site_name='ee_site',
                 stiffness=None,
                 damping=None,
                 num_waypoints=2,
                 context_dim=6,
                 latent_parameter_dim=6,
                 parameter_dim=8,
                 action_dim=6,
                 stiffness_high=None,
                 stiffness_low=None,
                 stiffness_initial=None,
                 weight_initial=None,
                 null_space_damping=3.0,
                 controlled_joints=None,
                 in_ee_frame=False
                 ):
        super(iMOGVIC, self).__init__(sim)
        self.sim = sim
        
        # Create a model for control
        # model_path = os.path.join(kuka_asset_dir(), model_path)
        model_path = os.path.join(model_root, model_path)
        self.model = mujoco_py.load_model_from_path(model_path)
        
        self.mujoco_config_kuka = MujocoConfig(model_path)
        
        self.in_ee_frame = in_ee_frame
        self.eef_name = site_name
        
        # Construct the action space in cartesian space.
        high_pos = pos_limit * np.ones(3)
        low_pos = -high_pos
        
        high_rot = rot_limit * np.ones(3)
        low_rot = -high_rot
        
        high = np.concatenate((high_pos, high_rot))
        low = np.concatenate((low_pos, low_rot))
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        
        # control scale
        self.scale = np.ones(6)
        self.scale[:3] *= pos_scale
        self.scale[3:6] *= rot_scale
        
        self.site_name = site_name
        self.pos_set = np.zeros(3)
        self.quat_set = identity_quat.copy()
        self.ori_set = None
        
        if stiffness is None:
            self.stiffness = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
        else:
            self.stiffness = np.ones(6) * stiffness
        
        if damping == 'auto':
            self.damping = 2 * np.sqrt(self.stiffness)
        else:
            # self.damping = np.ones(6) * damping
            # self.damping = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
            self.damping = 2 * np.sqrt(self.stiffness)
        
        self.context_dim = context_dim
        self.latent_parameter_dim = latent_parameter_dim
        self.parameter_dim = parameter_dim
        self.action_dim = action_dim
        
        self.stiffness_high = stiffness_high
        self.stiffness_low = stiffness_low
        self.stiffness_initial = stiffness_initial
        self.weight_initial = weight_initial
        self.num_waypoints = num_waypoints
        
        self.null_space_damping = null_space_damping
        
        # if controlled_joints is not None:
        self.sim_qpos_idx = get_qpos_indices(self.model, controlled_joints)
        self.sim_qvel_idx = get_qvel_indices(self.model, controlled_joints)
        self.sim_actuators_idx = get_actuator_indices(self.model, controlled_joints)
        self.sim_joint_idx = get_joint_indices(self.model, controlled_joints)
        
        self.self_qpos_idx = get_qpos_indices(self.model, controlled_joints)
        self.self_qvel_idx = get_qvel_indices(self.model, controlled_joints)
        self.self_actuators_idx = get_actuator_indices(self.model, controlled_joints)
        
        # robot states
        self.ee_pos = None
        self.ee_ori_mat = None
        self.ee_pos_vel = None
        self.ee_ori_vel = None
        self.joint_pos = None
        self.joint_vel = None
        
        # dynamics and kinematics
        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None
        
        # kp and kd
        kp = 150
        damping_ratio = 1
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio
        
        # initial parameters :::
        self.stiffness_list = np.ones((self.num_waypoints, 6))
        self.damping_list = np.ones((self.num_waypoints, 6))
        self.weights_list = np.ones((self.num_waypoints, 1))
        
        self.reference_list = np.zeros((self.num_waypoints, 6))
        self.ori_reference_list = np.zeros((self.num_waypoints, 6))
        self.optimal_pose = self.reference_list[0, :]

        self.stiffness_matrix = np.zeros(6)
        self.damping_matrix = np.zeros(6)
        self.V = None
    
    def update_state(self):
        """
            update state
        """
        # Only run update if self.new_update or force flag is set
        self.sim.forward()
        
        self.qpos_index = self.sim_qpos_idx
        self.qvel_index = self.sim_qvel_idx
        
        # return ee_pose
        self.ee_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)])
        self.ee_ori_mat = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.eef_name)].reshape([3, 3]))
        self.ee_pos_vel = np.array(self.sim.data.site_xvelp[self.sim.model.site_name2id(self.eef_name)])
        self.ee_ori_vel = np.array(self.sim.data.site_xvelr[self.sim.model.site_name2id(self.eef_name)])
        
        # return joint pose and velocity
        self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
        self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])
        
        # return jacobian matrix
        self.J_pos = np.array(self.sim.data.get_site_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_ori = np.array(self.sim.data.get_site_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))
        
        # return mass_matrix
        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
        
        # mass matrix in task space ：：
        self.M_xx = np.linalg.inv(self.J_full.dot(np.linalg.inv(self.mass_matrix)).dot(self.J_full.T))
        
        # ee_pose
        self.ee_pose = np.concatenate((np.array(self.ee_pos), transforms3d.euler.mat2euler(self.ee_ori_mat)))
        self.ee_pose_vel = np.concatenate((self.ee_pos_vel, self.ee_ori_vel), axis=0)
    
    def update_vic_torque(self):
        """
            get vic torque
        """
        self.update_state()
        
        # velocity error
        vel_pos_error = -self.ee_pos_vel
        vel_ori_error = -self.ee_ori_vel
        
        # position diff : current_state - optimal_state
        self.state = self.get_current_state(self.ee_pose)
        
        # print("ee_pose :::", self.ee_pose)
        # print("state :::", self.state)
        # print("optimal_pose :::", self.optimal_pose)

        # print("=================================================================================")
        # a function of state
        self.omega_weights, self.beta_weights = \
            self.get_non_linear_weight(state=self.state, stiffness=self.stiffness_list, weights=self.weights_list)
        
        self.stiffness_matrix, self.damping_matrix = \
            self.get_stiffness_damping(self.omega_weights, self.stiffness_list, self.damping_list)
        
        # force and torque of base component
        desired_force_list = self.omega_weights[0] * (
                np.multiply(-1 * np.array(self.state[:3]), np.array(self.stiffness_list[0, 0:3]))
                + np.multiply(vel_pos_error, self.damping_list[0, 0:3]))
        desired_torque_list = self.omega_weights[0] * (
                np.multiply(-1 * np.array(self.state[3:]), np.array(self.stiffness_list[0, 3:6]))
                + np.multiply(vel_ori_error, self.damping_list[0, 3:6]))
        
        # torque_weight_list = self.omega_weights
        torque_weight_list = self.weights_list
        
        # other components
        # for i in range(1, self.num_waypoints):
        #     position_error_point = self.reference_list[i, :3] - self.state[:3]
        #     desired_force_list += torque_weight_list[i] * (
        #             np.multiply(np.array(position_error_point), np.array(self.stiffness_list[i, 0:3]))
        #             + np.multiply(vel_pos_error, self.damping_list[i, 0:3]))
        #
        #     # ori_error_point = orientation_error(euler2mat(self.reference_list[i, 3:]), euler2mat(self.state[3:]))
        #     real_ori_attractor = transforms3d.euler.euler2mat(
        #         self.state[3],
        #         self.state[4],
        #         self.state[5],
        #         'sxyz')
        #     ref_ori_attractor = transforms3d.euler.euler2mat(
        #         self.reference_list[i, 3],
        #         self.reference_list[i, 4],
        #         self.reference_list[i, 5],
        #         'sxyz')
        #     ori_error_point = orientation_error(ref_ori_attractor, real_ori_attractor)
        #     desired_torque_list += torque_weight_list[i] * (
        #             np.multiply(np.array(ori_error_point), np.array(self.stiffness_list[i, 3:6]))
        #             + np.multiply(vel_ori_error, self.damping_list[i, 3:6]))
        
        desired_force_attractors, desired_torque_attractors = \
            self.get_attractor_force(torque_weight_list, vel_pos_error, vel_ori_error)
        desired_force_list += desired_force_attractors
        desired_torque_list += desired_torque_attractors
        
        # pos_error = self.ee_pos - self.reference_list[0, :3]
        # ori_error = orientation_error(self.ee_ori_mat, euler2mat(self.reference_list[0, 3:]))
        # state_pose = np.concatenate((pos_error, np.array(ori_error)), axis=0)
        
        self.V = self.energy_function(
            state_pos=self.state,
            state_vel=self.ee_pose_vel,
            stiffness=self.stiffness_list,
            weights=self.weights_list,
            beta_weights=self.beta_weights,
            M_xx=self.M_xx
        )
        
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix \
            = opspace_matrices(self.mass_matrix,
                               self.J_full,
                               self.J_pos,
                               self.J_ori)
        
        # Decouples desired positional control from orientation control
        desired_wrench = np.concatenate([desired_force_list, desired_torque_list])
        decoupled_wrench = np.dot(lambda_full, desired_wrench)
        
        # Gamma (without null torques) = J^T * F + gravity compensations
        self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation
        
        self.initial_joint = self.joint_pos
        self.torques += nullspace_torques(self.mass_matrix, nullspace_matrix,
                                          self.initial_joint, self.joint_pos, self.joint_vel)
        
        return self.torques, self.V, self.state, self.ee_pose_vel, self.stiffness_matrix, self.damping_matrix
    
    def get_current_state(self, state):
        """
            get current normalized state
        """
        current_state = np.zeros(6)
        current_state[:3] = state[:3] - self.optimal_pose[:3]
        current_euler = state[3:].copy()
        
        real_mat = transforms3d.euler.euler2mat(current_euler[0],
                                                current_euler[1],
                                                current_euler[2],
                                                'sxyz')
        optimal_mat = transforms3d.euler.euler2mat(self.optimal_pose[3],
                                                   self.optimal_pose[4],
                                                   self.optimal_pose[5],
                                                   'sxyz')
        
        current_state[3:] = orientation_error(real_mat, optimal_mat).copy()
        # current_state[3:] = orientation_error(optimal_mat, real_mat)
        return current_state.copy()
    
    def get_attractor_force(self, torque_weight_list,
                            vel_pos_error, vel_ori_error):
        desired_force_list = np.zeros(3)
        desired_torque_list = np.zeros(3)
        # calculate other points :
        for i in range(1, self.num_waypoints):
            position_error_point = self.reference_list[i, :3] - self.state[:3]
            desired_force_list = torque_weight_list[i] * (
                    np.multiply(np.array(position_error_point), np.array(self.stiffness_list[i, 0:3]))
                    + np.multiply(vel_pos_error, self.damping_list[i, 0:3]))
            
            real_ori_attractor = transforms3d.euler.euler2mat(
                self.ee_pose[3],
                self.ee_pose[4],
                self.ee_pose[5],
                'sxyz')
            ref_ori_attractor = transforms3d.euler.euler2mat(
                self.ori_reference_list[i, 3],
                self.ori_reference_list[i, 4],
                self.ori_reference_list[i, 5],
                'sxyz')
            ori_error_point = orientation_error(ref_ori_attractor, real_ori_attractor)
            desired_torque_list = torque_weight_list[i] * (
                    np.multiply(np.array(ori_error_point), np.array(self.stiffness_list[i, 3:6]))
                    + np.multiply(vel_ori_error, self.damping_list[i, 3:6]))
            
        return desired_force_list, desired_torque_list
    
    def set_params_direct(self, stiffness_list, damping_list, weights_list):
        self.stiffness_list = stiffness_list
        self.damping_list = damping_list
        self.weights_list = weights_list
    
    def set_way_points(self, reference_list):
        """
            Set task space set point list.
        """
        self.ori_reference_list = self.reference_list
        self.optimal_pose = reference_list[0, :]
        mean_set_list = []
        for i in range(self.num_waypoints):
            mean_set_list.append(self.get_current_state(reference_list[i, :]))
        self.reference_list = np.array(mean_set_list)

    def get_non_linear_weight(self, state, stiffness, weights):
        """
            get non-linear weights of DS
        """
        self.alpha_weights = np.zeros_like(weights)
        self.beta_weights = np.zeros_like(weights)
        self.omega_weights = np.zeros_like(weights)
        self.omega_weights[0] = 1
    
        if self.num_waypoints > 1:
            for i in range(1, weights.shape[0]):
                delta_pos = state[:3] - 1.1 * self.reference_list[i, :3]

                # mat_t = transforms3d.euler.euler2mat(state[3], state[4], state[5], 'sxyz')
                # mat_d = transforms3d.euler.euler2mat(2 * self.reference_list[i, 3],
                #                                      2 * self.reference_list[i, 4],
                #                                      2 * self.reference_list[i, 5], 'sxyz')
                # delta_ori = orientation_error(mat_d, mat_t)
                curr_ori_mat = transforms3d.euler.euler2mat(self.ee_pose[3],
                                                            self.ee_pose[4],
                                                            self.ee_pose[5],
                                                            'sxyz')

                attractor_ori_mat = transforms3d.euler.euler2mat(self.ori_reference_list[i, 3],
                                                                 self.ori_reference_list[i, 4],
                                                                 self.ori_reference_list[i, 5],
                                                                 'sxyz')
                
                optimal_mat = transforms3d.euler.euler2mat(self.optimal_pose[3],
                                                           self.optimal_pose[4],
                                                           self.optimal_pose[5],
                                                           'sxyz')
                
                delta_ori = orientation_error(curr_ori_mat, attractor_ori_mat) + \
                            orientation_error(optimal_mat, attractor_ori_mat)
                delta_pose = np.concatenate((delta_pos, np.zeros(3)), axis=0)
                
                # print("state :", orientation_error(curr_ori_mat, optimal_mat),
                #       orientation_error(curr_ori_mat, attractor_ori_mat),
                #       orientation_error(optimal_mat, attractor_ori_mat))
                # print("delta_pose :", delta_pose)
                # print('middle :', (state.dot(stiffness[i, :] * np.eye(6)).dot(delta_pose)))
                if (state.dot(stiffness[i, :] * np.eye(6)).dot(delta_pose)) >= 0:
                    self.alpha_weights[i] = state.dot(stiffness[i, :] * np.eye(6)).dot(delta_pose)
                    # print("alpha_weights :::", self.alpha_weights[i])
                else:
                    self.alpha_weights[i] = 0
            
                self.beta_weights[i] = np.exp(- 1 * (weights[i] / 4) * np.square(self.alpha_weights[i]))
                self.omega_weights[i] = self.alpha_weights[i] * self.beta_weights[i]
                # print("omega_weights :::", self.omega_weights[i])
        return self.omega_weights, self.beta_weights
    
    def set_action(self, action):
        '''
            Set setpoint
        '''
        action = action * self.scale
        
        dx = action[0:3].astype(np.float64)
        dr = action[3:6].astype(np.float64)
        
        pos, mat = forwardKinSite(self.sim, self.site_name, recompute=False)
        quat = mat2Quat(mat)
        
        if self.in_ee_frame:
            dx = mat.dot(dx)
        
        self.pos_set = pos + dx
        self.quat_set = quatAdd(quat, dr)

    def set_params(self, w=None):
        """
            reset iMOGVIC parameters
        """
        scale = 2.0
        if w is not None:
            stiffness_pos = []
            for i in range(self.num_waypoints):
                stiffness_pos.append(w[self.num_waypoints + i:self.num_waypoints + i + 3])
        
            self.weights_list = np.array(w[:self.num_waypoints])
            self.weights_list[0] = 1
            scale_stiffness_rotation = 5.0
            stiffness_rotation = np.ones((self.num_waypoints, 3)) * scale_stiffness_rotation
        
            self.stiffness_list = np.concatenate((stiffness_pos, stiffness_rotation), axis=1)
        
            self.damping_list = scale * np.sqrt(self.stiffness_list)
    
        print("stiffness_list", self.stiffness_list)
        print("damping_list", self.damping_list)
        print("weights_list", self.weights_list)

    def get_torque_reset(self, site_name):
        self.update_state()
    
        # desired_pos = None
        # Only linear interpolator is currently supported
        # desired_pos = reference_list[0, :3]
        # desired_ori = transforms3d.euler.euler2mat(reference_list[0, 3], reference_list[0, 4], reference_list[0, 5], 'sxyz')
        desired_pos = self.pos_set
        desired_ori = self.ori_set
    
        # # print("ori_mat :::", self.ee_ori_mat)
        # ori_error = orientation_error(desired_ori, self.ee_ori_mat)
        #
        # # print("target_rot :::", transforms3d.euler.mat2euler(desired_ori))
        # # print("real_rot :::", transforms3d.euler.mat2euler(self.ee_ori_mat))
        # # print("ori_error :::", ori_error)
        #
        # # Compute desired force and torque based on errors
        # position_error = desired_pos - self.ee_pos
        # # print("position_error :::", position_error)
        #
        # vel_pos_error = -self.ee_pos_vel
        #
        # # F_r = kp * pos_err + kd * vel_err
        # desired_force = (np.multiply(np.array(position_error), np.array(self.stiffness_list[0, 0:3]))
        #                  + np.multiply(vel_pos_error, self.damping_list[0, 0:3]))
        #
        # vel_ori_error = -self.ee_ori_vel
        # # print("vel_ori_error :::", vel_ori_error)
        # # print("desired_ori :::", desired_ori)
        #
        # # Tau_r = kp * ori_err + kd * vel_err
        # desired_torque = (np.multiply(np.array(ori_error), np.array(self.stiffness_list[0, 3:6]))
        #                   + np.multiply(vel_ori_error, self.damping_list[0, 3:6]))
        # # print("desired_torque ::", desired_torque)
        #
        # # print("J_full shape :::", self.J_full.shape)
        # # print("J_pos shape :::", self.J_pos.shape)
        # # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        # lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(self.mass_matrix,
        #                                                                          self.J_full,
        #                                                                          self.J_pos,
        #                                                                          self.J_ori)
        #
        # # Decouples desired positional control from orientation control
        # desired_wrench = np.concatenate([desired_force, desired_torque])
        # decoupled_wrench = np.dot(lambda_full, desired_wrench)
        #
        # # Gamma (without null torques) = J^T * F + gravity compensations
        # self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation
        #
        # # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        # #                     to the initial joint positions
        # self.torques += nullspace_torques(self.mass_matrix, nullspace_matrix,
        #                                   self.joint_pos, self.joint_pos, self.joint_vel)
    
        # print("torque command 1 :::", self.torques)
        # print("mass matrix :::", self.M_xx)
        #
    
        # pos, mat = forwardKinSite(self.sim, site_name, recompute=False)
        # desired_pos = reference_list[0, :3]
    
        pos, mat = self.get_pose_site(site_name)
        dx = desired_pos - pos
    
        # dr = subQuat(transforms3d.euler.euler2quat(reference_list[0, 3], reference_list[0, 4], reference_list[0, 5], 'sxyz'),
        #              transforms3d.quaternions.mat2quat(mat))
        dr = orientation_error(transforms3d.quaternions.quat2mat(self.quat_set), mat)
        # print("dr :::", dr)
    
        # dr_euler = reference_list[0, 3:] - transforms3d.euler.mat2euler(mat, 'sxyz')
        # print("dr_euler :::", dr_euler)
    
        # desired_ori = transforms3d.euler.euler2mat(reference_list[0, 3], reference_list[0, 4], reference_list[0, 5], 'sxyz')
        # dr_new = orientation_error(desired_ori, mat)
        # print("dr_new :::", dr_new)
    
        # dr = orientation_error(transforms3d.quaternions.quat2mat(self.quat_set), mat)
        dframe = np.concatenate((dx, dr))
    
        # dframe = np.concatenate((position_error, ori_error), axis=0)
    
        # Compute generalized forces from a virtual external force.
        jpos, jrot = forwardKinJacobianSite(self.sim, site_name, recompute=False)
        J = np.vstack((jpos[:, self.sim_qvel_idx], jrot[:, self.sim_qvel_idx]))
        cartesian_acc_des = self.stiffness * dframe - self.damping * J.dot(self.sim.data.qvel[self.sim_qvel_idx])
        impedance_acc_des = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-6 * np.eye(6), cartesian_acc_des))
    
        # Add damping in the null space of the the Jacobian
        projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T), J))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_vel = projection_matrix.dot(self.sim.data.qvel[self.sim_qvel_idx])
        impedance_acc_des += -self.null_space_damping * null_space_vel  # null space damping
    
        # print("impedance_acc_des :::", impedance_acc_des)
        # Cancel other dynamics and add virtual damping using inverse dynamics.
        acc_des = np.zeros(self.sim.model.nv)
        acc_des[self.sim_qvel_idx] = impedance_acc_des
        self.sim.data.qacc[:] = acc_des
        mujoco_py.functions.mj_inverse(self.model, self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[self.sim_actuators_idx].copy()
    
        # print("torque command 2", id_torque)
        return id_torque
    
        # return self.torques

    def get_euler_torque(self, reference_list):
        # optimal pose
        self.optimal_pose = reference_list[0, :]
    
        # current state
        state = np.concatenate((self.ee_pos, transforms3d.euler.mat2euler(self.ee_ori_mat)), axis=0) \
                - self.optimal_pose
    
        # attractor point transform
        attractor_points_pose = reference_list - self.optimal_pose
    
        self.omega_weights = np.zeros_like(self.weights_list)
        self.alpha_weights = np.zeros_like(self.weights_list)
        self.beta_weights = np.zeros_like(self.weights_list)
        self.omega_weights[0] = 1
    
        if self.num_waypoints > 1:
            for i in range(1, self.weights_list.shape[0]):
                delta_pose = state - 2 * attractor_points_pose[i, :]
                if (state.dot(self.stiffness_list[i, :] * np.eye(6)).dot(delta_pose)) >= 0:
                    self.alpha_weights[i] = state.dot(self.stiffness_list[i, :] * np.eye(6)).dot(delta_pose)
                else:
                    self.alpha_weights[i] = 0
                self.beta_weights[i] = np.exp(- 1 * (self.weights_list[i] / 4) * np.square(self.alpha_weights[i]))
                self.omega_weights[i] = self.alpha_weights[i] * self.beta_weights[i]
        print("weights :::", self.omega_weights)
    
        self.omega_weights[1] = 1.0
        cartesian_acc_des = np.zeros(6)
        # calculate difference with attractor points
        for i in range(self.num_waypoints):
            # dx = state_pos - attractor_points_pos[i]
            # # print('pos_list :::', self.pos_set_list[i])
            # # print('quat_list :::', self.quat_set_list[i])
            # dr = get_orientation_error(self.quat_optimal_point, state_quat)
            # print("dr_1 ::::", dr)
            # # dr = - subQuat(self.quat_optimal_point, state_quat)
            # # print("dr ::::", dr)
            # # dr = state_quat - attractor_points_quat[i]
            dframe = state - attractor_points_pose[i, :]
        
            # Compute generalized forces from a virtual external force.
            cartesian_acc_des += - self.omega_weights[i] * self.stiffness_list[i] * dframe \
                                 - self.omega_weights[i] * self.damping_list[i] * self.J_full.dot(
                self.sim.data.qvel[self.sim_qvel_idx])
    
        impedance_acc_des = self.J_full.T.dot(
            np.linalg.solve(self.J_full.dot(self.J_full.T) + 1e-6 * np.eye(6), cartesian_acc_des))
    
        # Add damping in the null space of Jacobian
        projection_matrix = self.J_full.T.dot(np.linalg.solve(self.J_full.dot(self.J_full.T), self.J_full))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_vel = projection_matrix.dot(self.sim.data.qvel[self.sim_qvel_idx])
        impedance_acc_des += -self.null_space_damping * null_space_vel
    
        # Cancel other dynamics and add virtual damping using inverse dynamics.
        acc_des = np.zeros(self.sim.model.nv)
        acc_des[self.sim_qvel_idx] = impedance_acc_des
        self.sim.data.qacc[:] = acc_des
        mujoco_py.functions.mj_inverse(self.model, self.sim.data)
    
        # print("sim_actuators_idx :::", self.sim_actuators_idx)
        id_torque = self.sim.data.qfrc_inverse[self.sim_actuators_idx].copy()
        # final_torque = np.concatenate([id_torque, np.array([0., 0.])])
    
        return id_torque

    def update_initial_joints(self, initial_joints):
        """
            Updates the internal attribute self.initial_joints. This is useful for updating changes in controller-specific
            behavior, such as with OSC where self.initial_joints is used for determine nullspace actions

            This function can also be extended by subclassed controllers for additional controller-specific updates

            Args:
                initial_joints (Iterable): Array of joint position values to update the initial joints
        """
        self.initial_joint = np.array(initial_joints)
        self.update_state()
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat

    def set_target(self, pos, quat):
        '''
            Set optimal target
        '''
        self.pos_set = pos
        self.quat_set = quat
        self.ori_set = transforms3d.quaternions.quat2mat(self.quat_set)
    
    def get_old_iros_torque(self):
        """ get iros torque """
        pos, mat = forwardKinSite(self.sim, self.site_name, recompute=False)
        print("actual_pos", pos)
        quat = mat2Quat(mat)
        print("target_pos", self.pos_set)
        dx = self.pos_set - pos
        dr = subQuat(self.quat_set, quat)
        dr = np.array([0.0, 0.0, 0.0])
        print("dr :::", dr)
        dframe = np.concatenate((dx, dr))
        print("actual_quat", quat)
        print("target_quat", self.quat_set)
        
        # Compute generalized forces from a virtual external force.
        jpos, jrot = forwardKinJacobianSite(self.sim, self.site_name, recompute=False)
        J = np.vstack((jpos[:, self.sim_qvel_idx], jrot[:, self.sim_qvel_idx]))
        cartesian_acc_des = self.stiffness * dframe - self.damping * J.dot(self.sim.data.qvel[self.sim_qvel_idx])
        # print("cartesian acc :::", cartesian_acc_des)
        # print("stiffness :::", self.stiffness)
        # print("damping :::", self.damping)
        impedance_acc_des = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-6 * np.eye(6), cartesian_acc_des))
        
        # Add damping in the null space of the the Jacobian
        projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T), J))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_vel = projection_matrix.dot(self.sim.data.qvel[self.sim_qvel_idx])
        impedance_acc_des += -self.null_space_damping * null_space_vel  # null space damping
        
        # Cancel other dynamics and add virtual damping using inverse dynamics.
        acc_des = np.zeros(self.sim.model.nv)
        acc_des[self.sim_qvel_idx] = impedance_acc_des
        self.sim.data.qacc[:] = acc_des
        mujoco_py.functions.mj_inverse(self.model, self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[self.sim_actuators_idx].copy()
        
        final_torque = np.concatenate([id_torque, np.array([0., 0.])])
        
        return final_torque
    
    def get_torque(self):
        '''
            Update the impedance control setpoint and compute the torque.
        '''
        torque, V, pose_err, vel_err, stiffness_eqv, damping_eqv = self.update_vic_torque()
        return torque
    
    def energy_function(self, state_pos=None, state_vel=None,
                        stiffness=None, weights=None,
                        beta_weights=None, M_xx=None
                        ):
        """
            Lyapunov function :::
        """
        # M_xx = np.linalg.inv(J.dot(np.linalg.inv(M_qq)).dot(J.T))
        # print("M_xx :::", M_xx.shape)
        # print("weights :::", weights)
        # print("beta :::", beta_weights)
        # print("state_pos :::", state_pos)
        # print("state_vel :::", state_vel)
        # print("Calculatin_1 :::", state_vel.dot(M_xx).dot(state_vel))
        # print("Calculatin_2 :::", (1/weights[1:] * (1 - beta_weights[1:])).sum())
        # print("Calculatin_3 :::", state_vel.dot(M_xx).dot(state_vel))
        
        # state_vel[3:] = np.zeros(3)
        
        # state_vel[1] = 0
        # V_1 = 1 / 2 * state_pos.dot(stiffness[0, :3] * np.eye(3)).dot(state_pos)
        # V_2 = (1 / weights[1:] * (1 - beta_weights[1:])).sum()
        # V_3 = 1 / 2 * state_vel.dot(M_xx).dot(state_vel)
        # V = V_1 + V_2 + V_3
        
        V_1 = 1 / 2 * state_pos.dot(stiffness[0, :] * np.eye(6)).dot(state_pos)
        V_2 = (1 / weights[1:] * (1 - beta_weights[1:])).sum()
        V_3 = 1 / 2 * state_vel.dot(M_xx).dot(state_vel)
        # V = V_1 + V_2 + V_3
        V = V_1 + V_2 + V_3
        
        return V
    
    def get_stiffness_damping(self, omega_weights, stiffness, damping):
        """ calculate stiffness and damping matrix """
        for i in range(omega_weights.shape[0]):
            self.stiffness_matrix += omega_weights[i] * stiffness[i, :]
            self.damping_matrix += omega_weights[i] * damping[i, :]
        
        return self.stiffness_matrix, self.damping_matrix
    
    def get_pose_site(self, site_name):
        # return ee_pose
        site_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(site_name)])
        site_ori_mat = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(site_name)].reshape([3, 3]))
        
        site_pos_vel = np.array(self.sim.data.site_xvelp[self.sim.model.site_name2id(site_name)])
        site_ori_vel = np.array(self.sim.data.site_xvelr[self.sim.model.site_name2id(site_name)])
        
        return site_pos, site_ori_mat
    
    def get_robot_joints(self):
        self.update_state()
        return self.joint_pos, self.joint_vel
    
    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")
        
        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums
    
    @property
    def torque_compensation(self):
        """
        Gravity compensation for this robot arm

        Returns:
            np.array: torques
        """
        return self.sim.data.qfrc_bias[self.qvel_index]
    
    @property
    def actuator_limits(self):
        """
        Torque limits for this controller

        Returns:
            2-tuple:

                - (np.array) minimum actuator torques
                - (np.array) maximum actuator torques
        """
        return self.actuator_min, self.actuator_max
    
    @property
    def control_limits(self):
        """
        Limits over this controller's action space, which defaults to input min/max

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        return self.input_min, self.input_max
    
    def get_info(self, state_pos, state_vel, J):
        """ get energy info for evaluation"""
        
        M_qq = self.mujoco_config_kuka.M(q=self.sim.data.qpos[self.sim_qpos_idx])
        
        # calculate energy function
        self.V = self.energy_function(
            state_pos=np.array(state_pos),
            state_vel=np.array(state_vel),
            stiffness=self.stiffness_list,
            weights=self.weights_list,
            beta_weights=self.beta_weights,
            M_qq=M_qq,
            J=J
        )
        
        # return equvilent stiffness and damping
        self.stiffness_eqv, self.damping_eqv = \
            self.get_stiffness_damping(omega_weights=self.omega_weights, stiffness=self.stiffness_list,
                                       damping=self.damping_list)
    
    def get_impedance_policy(self, cartesian_acc_des, state_pos, state_quat, J,
                             stiffness, damping, omega_weights):
        
        # cartesian_acc_des = np.zeros(6)
        # calculate difference with attractor points
        for i in range(self.num_waypoints):
            # i = 0 , S0 and D0
            dx = state_pos - self.pos_set_list[i]
            dr = quat2Vel(state_quat)
            dframe = np.concatenate((dx, dr))
            # print('Difference dframe :::', dframe)
            
            # Compute generalized forces from a virtual external force.
            cartesian_acc_des += - omega_weights[i] * stiffness[i] * dframe - \
                                 omega_weights[i] * damping[i] * J.dot(self.sim.data.qvel[self.sim_qvel_idx])
        
        return cartesian_acc_des
    
    # def set_params(self, w=None):
    # 	""" reset controller parameters """
    # 	if w is not None:
    # 		if self.num_waypoints == 3:
    # 			weights = np.array(w[:3])
    # 			stiffness_pos = np.array([w[3:6], w[6:9], w[9:]])
    # 			stiffness_rotation = np.ones((3, 3))
    # 			stiffness = np.concatenate((stiffness_pos, stiffness_rotation), axis=1)
    # 			damping = 2 * np.sqrt(stiffness)
    # 			weights = weights
    # 			weights[0] = 1
    # 		else:
    # 			weights = np.array(w[:2])
    # 			stiffness_pos = np.array([w[2:5], w[5:8]])
    # 			scale_stiffness_rotation = 3.0
    # 			stiffness_rotation = np.ones((2, 3)) * scale_stiffness_rotation
    # 			stiffness = np.concatenate((stiffness_pos, stiffness_rotation), axis=1)
    # 			damping = 2 * np.sqrt(stiffness)
    # 			weights = weights
    # 			weights[0] = 1
    #
    # 		self.stiffness_list = stiffness
    # 		self.damping_list = damping
    # 		self.weights_list = weights
    # 		# print("stiffness_list", self.stiffness_list)
    # 		# print("self.damping_list", self.damping_list)
    # 		# print("self.weights_list", self.weights_list)


register_controller(iMOGVIC, "iMOGVIC")
