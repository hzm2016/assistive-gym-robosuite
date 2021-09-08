import os
import numpy as np

from envs.gym_kuka_mujoco.envs import kuka_env
from envs.gym_kuka_mujoco.envs.assets import kuka_asset_dir
from envs.gym_kuka_mujoco.utils.kinematics import forwardKinSite, forwardKinJacobianSite
from envs.gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from envs.gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat
from envs.gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from envs.gym_kuka_mujoco.utils.control_utils import *
from envs.gym_kuka_mujoco.utils.transform_utils import *

import transforms3d


class PercisionPegInsertionEnv(kuka_env.KukaEnv):
	"""
		Peg in hole environment
	"""
	def __init__(self,
				 *args,
				 hole_id=99,
				 gravity=True,
				 obs_scaling=0.1,
				 sample_good_states=False,
				 use_ft_sensor=False,
				 use_rel_pos_err=False,
				 quadratic_cost=True,
				 quadratic_rot_cost=True,
				 regularize_pose=False,
				 linear_cost=False,
				 logarithmic_cost=False,
				 sparse_cost=False,
				 observe_joints=True,
				 contextual_policy=True,
				 in_peg_frame=False,
				 max_episode_steps=400,
	             model_root=None,
				 model_path="/single_peg_hole/full_peg_insertion_experiment_no_gravity_moving_hole_id=025.xml",
				 random_hole_file='random_reachable_holes_small_randomness.npy',
				 init_randomness=0.01,
				 sac_reward_scale=1.0,
				 **kwargs):
		
		# Store arguments.
		self.obs_scaling = obs_scaling
		self.sample_good_states = sample_good_states
		self.use_ft_sensor = use_ft_sensor
		self.use_rel_pos_err = use_rel_pos_err
		self.regularize_pose = regularize_pose
		self.quadratic_cost = quadratic_cost
		self.linear_cost = linear_cost
		self.logarithmic_cost = logarithmic_cost
		self.sparse_cost = sparse_cost
		self.quadratic_rot_cost = quadratic_rot_cost
		self.observe_joints = observe_joints
		self.in_peg_frame = in_peg_frame
		self.init_randomness = init_randomness
		self._max_episode_steps = max_episode_steps
		
		# ============================== context_dim ==============================
		self.contextual_policy = contextual_policy
		self.sac_reward_scale = sac_reward_scale
		# ============================== context_dim ==============================
		
		# Resolve the models path based on the hole_id.
		gravity_string = '' if gravity else '_no_gravity'
		if hole_id >= 0:
			kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment{}_moving_hole_id={:03d}.xml'.format(
				gravity_string, hole_id))
		else:
			kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_no_hole{}.xml').format(
				gravity_string)
		
		# kwargs['model_root'] = model_root
		kwargs['model_path'] = model_path
		super(PercisionPegInsertionEnv, self).__init__(*args, **kwargs)
		
		self.context_dim = self.controller.context_dim
		self.parameter_dim = self.controller.parameter_dim
		self.latent_parameter_dim = self.controller.latent_parameter_dim
		self.latent_parameter_low = self.controller.stiffness_low
		self.latent_parameter_high = self.controller.stiffness_high
		self.latent_parameter_initial = self.controller.stiffness_initial
		self.num_waypoints = self.controller.num_waypoints
		self.action_dim = self.controller.action_dim
		
		self.Q_pos = np.diag([100, 100, 100])
		self.Q_rot = np.diag([30, 30, 30])
		
		if self.regularize_pose:
			self.Q_pose_reg = np.eye(7)

		# Compute good states using inverse kinematics.
		if self.random_target:
			self.reachable_holes = np.load(os.path.join(kuka_asset_dir(), random_hole_file), allow_pickle=True)
			# self.hole_data = self.np_random.choice(self.reachable_holes)
			self.hole_data = self.reachable_holes[0]
			self._reset_target()
		else:
			self.reachable_holes = np.load(os.path.join(kuka_asset_dir(), random_hole_file), allow_pickle=True)
			print("reachable_holes :", self.reachable_holes[0])
			self.hole_data = self.reachable_holes[0]
			# 'hole_pos': array([0.80747761, 0.01258849, 1.24468107]), 'hole_quat': array(
			# 	[0.9997048, 0.01053176, 0.01507271, 0.01588089])
			self._reset_fix_target()
			# self.good_states = hole_insertion_samples(self.sim, range=[0., 0.06])
		
		self.distance_safe_threshold = 0.3
		self.distance_done_threshold = 0.05
	
	def _get_reward(self, state, action):
		'''
			Compute single step reward.
		'''
		# compute position and rotation error
		pos, rot = forwardKinSite(self.sim, ['peg_tip', 'hole_base'], recompute=False)
		# pos_err = pos[0] - pos[1]
		# rot_err = orientation_error(rot[0], rot[1])
		
		# dist = np.sqrt(pos_err.dot(pos_err))
		# peg_quat = mat2Quat(rot[0])
		# hole_quat = mat2Quat(rot[1])
		# rot_err = subQuat(peg_quat, hole_quat)
		# peg_quat = transforms3d.euler.mat2euler(rot[0], 'sxyz')
		# hole_quat = transforms3d.euler.mat2euler(rot[1], 'sxyz')
		
		pos_err = self.controller.state[:3]
		rot_err = self.controller.state[3:]
		
		pose_err = self.sim.data.qpos - self.good_states[0]
		
		peg_tip_id = self.model.site_name2id('peg_tip')
		jacp, jacv = forwardKinJacobianSite(self.sim, peg_tip_id, recompute=False)
		peg_tip_vel = jacp.dot(self.data.qvel[:])
		
		# quadratic cost on the error and action
		# rotate the cost terms to align with the hole
		Q_pos = rotate_cost_by_matrix(self.Q_pos, rot[1].T)
		# Q_vel = rotate_cost_by_matrix(self.Q_vel,rot[1].T)
		Q_rot = self.Q_rot
		
		reward_info = dict()
		reward = 0.0

		# reward_info['quaternion_reward'] = -rot_err.dot(Q_rot).dot(rot_err)
		if self.quadratic_rot_cost:
			reward_info['quadratic_orientation_reward'] = -rot_err.dot(Q_rot).dot(rot_err)
			reward += reward_info['quadratic_orientation_reward']
		
		if self.quadratic_cost:
			reward_info['quadratic_position_reward'] = -pos_err.dot(Q_pos).dot(pos_err)
			reward += reward_info['quadratic_position_reward']
		
		if self.linear_cost:
			reward_info['linear_position_reward'] = -np.sqrt(pos_err.dot(Q_pos).dot(pos_err))
			reward += reward_info['linear_position_reward']
		
		if self.logarithmic_cost:
			rew_scale = 2
			eps = 10.0 ** (-rew_scale)
			zero_crossing = 0.05
			reward_info['logarithmic_position_reward'] = -np.log10(
				np.sqrt(pos_err.dot(Q_pos).dot(pos_err)) / zero_crossing * (1 - eps) + eps)
			reward += reward_info['logarithmic_position_reward']
		
		if self.sparse_cost:
			reward_info['sparse_position_reward'] = 10.0 if np.sqrt(pos_err.dot(pos_err)) < 1e-2 else 0
			reward += reward_info['sparse_position_reward']
		
		if self.regularize_pose:
			reward_info['pose_regularizer_reward'] = -pose_err.dot(self.Q_pose_reg).dot(pose_err)
			reward += reward_info['pose_regularizer_reward']
		
		# reward_info['velocity_reward'] = -np.sqrt(peg_tip_vel.dot(Q_vel).dot(peg_tip_vel))
		# reward += reward_info['velocity_reward']
		
		return reward, reward_info
	
	def _get_info(self):
		"""
			get feedback
		"""
		info = dict()
		
		# pos, rot = forwardKinSite(self.sim, ['peg_tip', 'hole_base'], recompute=False)
		# pos_err = pos[0] - pos[1]
		# dist = np.sqrt(pos_err.dot(pos_err))
		dist = np.linalg.norm(self.controller.state, ord=2)
		
		info['tip_distance'] = dist
		info['success'] = float(dist < self.distance_done_threshold)
		# info['qpos'] = self.data.qpos.copy()
		# info['qvel'] = self.data.qvel.copy()
		info['stiffness_eqv'] = self.controller.stiffness_matrix
		info['damping_eqv'] = self.controller.damping_matrix
		info['energy'] = self.controller.V
		
		return info
	
	def _get_done(self):
		"""
			judge distance
		"""
		# some problem
		# target_pos_base, target_quat_base, target_euler_base = \
		# 	self._get_site_cartesian_pose('hole_base')
		# target_pos_tip, target_quat_tip, target_euler_tip = \
		# 	self._get_site_cartesian_pose('peg_tip')
		
		target_distance = np.linalg.norm(self.controller.state, ord=2)
		# print("target_distance :::", target_distance)
		if target_distance < self.distance_done_threshold:
			return True
		else:
			return False
	
	def _get_state_obs(self):
		'''
			Compute the observation at the current state.
		'''
		if self.observe_joints:
			obs = super(PercisionPegInsertionEnv, self)._get_state_obs()
		else:
			obs = np.zeros(0)
		
		# Return superclass observation stacked with the ft observation.
		if not self.initialized:
			ft_obs = np.zeros(6)
		else:
			# Compute F/T sensor data
			ft_obs = self.sim.data.sensordata
			# print("ft_obs :::", ft_obs)
			obs = obs/self.obs_scaling
		
		if self.use_ft_sensor:
			obs = np.concatenate([obs, ft_obs])
		
		# End effector position
		pos, rot = forwardKinSite(self.sim, ['peg_tip', 'hole_base', 'hole_top'])
		
		if self.use_rel_pos_err:
			pos_obs = pos[1] - pos[0]
			quat_peg_tip = mat2Quat(rot[0])
			quat_hole_base = mat2Quat(rot[1])
			rot_obs = subQuat(quat_hole_base, quat_peg_tip).copy()
			hole_top_obs = pos[2] - pos[0]
		else:
			# TODO: we probably also want the EE position in the world
			pos_obs = pos[1].copy()
			rot_obs = mat2Quat(rot[1])
			hole_top_obs = pos[2]
		
		# End effector velocity
		peg_tip_id = self.model.site_name2id('peg_tip')
		jacp, jacr = forwardKinJacobianSite(self.sim, peg_tip_id, recompute=False)
		peg_tip_lin_vel = jacp.dot(self.sim.data.qvel)
		peg_tip_rot_vel = jacr.dot(self.sim.data.qvel)
		
		# Transform into end effector frame
		if self.in_peg_frame:
			pos_obs = rot[0].T.dot(pos_obs)
			hole_top_obs = rot[0].T.dot(hole_top_obs)
			peg_tip_lin_vel = rot[0].T.dot(peg_tip_lin_vel)
			peg_tip_rot_vel = rot[0].T.dot(peg_tip_rot_vel)
		
		obs = np.concatenate([obs, pos_obs, rot_obs, peg_tip_lin_vel, peg_tip_rot_vel, hole_top_obs])
		return obs
	
	def _get_target_obs(self):
		# Compute relative position error
		
		# target_pos, target_rot = forwardKinSite(self.sim, ['peg_tip', 'hole_base', 'hole_top'])
		
		# quat_hole_top = mat2Quat(target_rot[1])
		# pos_hole_top = target_pos[1]
		
		# self.pos_optimal_point = target_pos[1]
		# self.quat_optimal_point = mat2Quat(target_rot[1])
		
		# if self.use_rel_pos_err:
		#     pos_obs = pos[1] - pos[0]
		#     quat_peg_tip = mat2Quat(rot[0])
		#     quat_hole_base = mat2Quat(rot[1])
		#     rot_obs = subQuat(quat_hole_base, quat_peg_tip).copy()
		#     hole_top_obs = pos[2] - pos[0]
		# else:
		#     pos_obs = pos[1].copy()
		#     rot_obs = mat2Quat(rot[1])
		#     hole_top_obs = pos[2]
		#
		# if self.in_peg_frame:
		#     pos_obs = rot[0].T.dot(pos_obs)
		#     hole_top_obs = rot[0].T.dot(hole_top_obs)
		#     peg_tip_lin_vel = rot[0].T.dot(peg_tip_lin_vel)
		#     peg_tip_rot_vel = rot[0].T.dot(peg_tip_rot_vel)
		
		# return target_pos, target_rot
		return np.zeros(0)
	
	def _reset_state(self):
		'''
			Reset the robot state and return the observation.
		'''
		qvel = np.zeros(7)

		if self.sample_good_states and self.np_random.uniform() < 0.5:
			qpos = self.np_random.choice(self.good_states)
			self.set_state(qpos, qvel)
			self.sim.forward()
		else:
			# qpos = self.good_states[-1] + self.np_random.uniform(-self.init_randomness, self.init_randomness, 7)
			# print("Good states :", self.good_states)
			qpos = self.good_states[0]
			# print("Reset state :", qpos)
			self.set_state(qpos, qvel)
			self.sim.forward()
			# while self.sim.data.ncon > 0:
			# qpos = self.good_states[-1] + self.np_random.uniform(-self.init_randomness, self.init_randomness, 7)
			# self.set_state(qpos, qvel)
			# self.sim.forward()
		
		# if self.random_initial:
		self.initial_offset = np.zeros(3)
		self.initial_offset[:2] = np.random.uniform(-0.035, 0.035, size=2)
		self.initial_offset[2] = 0.17
		
		pos_hole_top, quat_hole_top, _ = self._get_site_cartesian_pose('hole_top')
		# print("pose_hole_top", pos_hole_top)
		pos_hole_base, quat_hole_top, _ = self._get_site_cartesian_pose('hole_base')
		# print("pose_hole_base", pos_hole_base)
		pos_tip, quat_tip, euler_tip = self._get_site_cartesian_pose('peg_tip')
		# print("pose_tip :", pos_tip)
		
		""" move to initial state """
		self._move_to_target(pos_hole_top, quat_hole_top, self.initial_offset)
		
	def _reset_target(self):
		'''
			Resets the hole position
		'''
		# if self.contextual_policy is False:
		# 	self.hole_data = self.np_random.choice(self.reachable_holes)
		self.good_states = self.hole_data['good_poses']
		self.sim.data.set_mocap_pos('hole', self.hole_data['hole_pos'] + np.array([-0.1, 0.0, -0.2]))
		self.sim.data.set_mocap_quat('hole', self.hole_data['hole_quat'])
		# print("Reset Target" * 10)
		
	def _reset_fix_target(self):
		'''
			Resets the hole position
		'''
		if self.contextual_policy is False:
			self.hole_data = self.np_random.choice(self.reachable_holes)
		# print("Hole data :", self.hole_data)
		self.good_states = self.hole_data['good_poses']
		# print("Good states :", self.good_states)
		self.sim.data.set_mocap_pos('hole', self.hole_data['hole_pos'])
		self.sim.data.set_mocap_quat('hole', self.hole_data['hole_quat'])
		self.sim.forward()
		# print("Reset Fixed Target ::::::::::::::::::::::::::::")
	
	def get_context(self):
		"""
			Return the context represents ::: target hole position
		"""
		# if self.contextual_policy:
		# 	self.hole_data = self.np_random.choice(self.reachable_holes)
		# 	self._reset_target()
		# 	self.sim.forward()
		#
		# 	# some problem
		# 	target_pos_base, target_quat_base, target_euler_base = \
		# 		self._get_site_cartesian_pose('hole_base')
		#
		# 	target_pos_top, target_quat_top, target_euler_top = \
		# 		self._get_site_cartesian_pose('hole_top')
		#
		# 	target_pos_tip, target_quat_tip, target_euler_tip = \
		# 		self._get_site_cartesian_pose('ee_site')
		# 	print("peg_tip_pos :", target_pos_tip, "peg_tip_euler :", target_euler_tip)
		# 	print("peg_base_pos :", target_pos_base, "peg_euler_base :", target_euler_base)
		#
		# 	pos_error = np.array(target_pos_tip - target_pos_base)
		# 	context = np.concatenate((pos_error, target_euler_base))
		#
		# 	print("context :", context)
		# 	return context, target_pos_top, target_quat_base, target_euler_top
		# else:
		# 	# some problem
		# 	target_pos_base, target_quat_base, target_euler_base = \
		# 		self._get_site_cartesian_pose('hole_base')
		#
		# 	target_pos_top, target_quat_top, target_euler_top = \
		# 		self._get_site_cartesian_pose('hole_top')
		#
		# 	target_pos_tip, target_quat_tip, target_euler_tip = \
		# 		self._get_site_cartesian_pose('ee_site')
		#
		# 	pos_error = np.array(target_pos_tip - target_pos_base)
		# 	context = np.concatenate((pos_error, target_euler_base))
		#
		# 	print("fixed context :", context)
		# 	return context, target_pos_top, target_quat_base, target_euler_top
		
		# some problem
		target_pos_base, target_quat_base, target_euler_base = \
			self._get_site_cartesian_pose('hole_base')
		
		target_pos_top, target_quat_top, target_euler_top = \
			self._get_site_cartesian_pose('hole_top')
		
		target_pos_tip, target_quat_tip, target_euler_tip = \
			self._get_site_cartesian_pose('peg_tip')
		
		# print("peg_tip_pos :", target_pos_tip, "peg_tip_euler :", target_euler_tip)
		# print("peg_base_pos :", target_pos_base, "peg_euler_base :", target_euler_base)
		# print("hole_top_pos :", target_pos_top, "hole_top_euler :", target_euler_top)
		
		pos_error = np.array(target_pos_tip - target_pos_base)
		context = np.concatenate((pos_error, target_quat_base))

		return context, target_pos_base, target_quat_base, target_euler_top
	
	def set_params(self, stiffness_list, damping_list, weights_list):
		self.controller.set_params_direct(stiffness_list, damping_list, weights_list)
	
	def set_waypoints(self, way_points_list):
		self.controller.set_way_points(way_points_list)
	
	# def set_waypoints(self):
	# 	"""
	# 		Set waypoints
	# 	"""
	# 	# target_pos, target_rot = forwardKinSite(self.sim, ['peg_tip', 'hole_base', 'hole_top'])
	# 	#
	# 	# self.pos_optimal_point = target_pos[1]
	# 	# self.quat_optimal_point = mat2Quat(target_rot[1])
	# 	#
	# 	# quat_hole_top = mat2Quat(target_rot[2])
	# 	# quat_hole_base = mat2Quat(target_rot[1])
	#
	# 	target_pos_base, target_quat_base, target_euler_base = self._get_site_cartesian_pose('hole_base')
	# 	# print("top_pos :::", target_pos)
	# 	# print("euler_base :::", target_euler)
	# 	target_pos_top, target_quat_top, target_euler_top = self._get_site_cartesian_pose('hole_top')
	# 	# print("euler_base :::", target_euler_base)
	# 	# print("top_pos :::", target_pos_base)
	# 	target_pos_tip, target_quat_tip, target_euler_tip = self._get_site_cartesian_pose('peg_tip')
	#
	# 	# print("euler_tip :::", target_euler_tip)
	# 	# if self.controller.num_waypoints == 3:
	# 	# 	self.pos_list = np.concatenate(([target_pos[1]], [target_pos[2]], [target_pos[2] - [0.0, 0.0, -0.05]]), axis=0)
	# 	# 	self.quat_list = np.concatenate(([quat_hole_base], [quat_hole_top], [quat_hole_top]), axis=0)
	# 	# else:
	# 	# 	self.pos_list = np.concatenate(([target_pos[2]], [target_pos[2]]), axis=0)
	# 	# 	self.quat_list = np.concatenate(([quat_hole_top], [quat_hole_top]), axis=0)
	#
	# 	self.pos_list = np.concatenate(([target_pos_base + [0.0, 0.0, 0.0]], [target_pos_top + [0.0, 0.0, 0.0]]), axis=0)
	# 	self.quat_list = np.concatenate(([target_euler_base], [target_euler_top]), axis=0)
	# 	way_points_list = np.concatenate((self.pos_list, self.quat_list), axis=1)
	# 	print("way_point_list :::", way_points_list)
	# 	self.controller.set_way_points(way_points_list)
	# 	print("reference list :::", self.controller.reference_list)
		
	def get_safe(self):
		""" used to judge the safety or not """
		target_pos, target_rot = forwardKinSite(self.sim, ['peg_tip', 'hole_base', 'hole_top'])
		target_distance = np.linalg.norm((target_pos[0] - target_pos[1]), ord=2)
		if target_distance > self.distance_safe_threshold:
			return False
		else:
			return True
