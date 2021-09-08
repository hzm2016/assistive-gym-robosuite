import os

import mujoco_py
import numpy as np
from gym import spaces

from envs.mujoco.controllers.base_controller import BaseController
from envs.mujoco.controllers.registry import register_controller
from envs.mujoco.envs.assets import kuka_asset_dir
from envs.mujoco.utils.kinematics import forwardKinSite, forwardKinJacobianSite, forwardVelKinSite
from envs.mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, \
	get_joint_indices
from envs.mujoco.utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat, quat2Vel

from envs.mujoco.mujoco_config import MujocoConfig


class RLVIC(BaseController):
	'''
		An inverse dynamics controller that used PD gains to compute a desired acceleration.
	'''
	def __init__(self,
				 sim,
				 pos_scale=1.0,
				 rot_scale=0.3,
				 pos_limit=1.0,
				 rot_limit=1.0,
				 model_path='full_kuka_no_collision_no_gravity.xml',
				 site_name='ee_site',
				 stiffness=None,
				 damping=None,
				 num_waypoints=3,
				 null_space_damping=1.0,
				 controlled_joints=None,
				 in_ee_frame=False):
		
		super(RLVIC, self).__init__(sim)
		self.sim = sim
		
		# Create a model for control
		model_path = os.path.join(kuka_asset_dir(), model_path)
		self.model = mujoco_py.load_model_from_path(model_path)
		
		self.mujoco_config_kuka = \
			MujocoConfig('/home/zhimin/code/5_tsinghua_assembly_projects/rl-robotic-assembly-control/envs/mujoco/envs/assets/full_kuka_mesh_collision.xml')
		
		self.in_ee_frame = in_ee_frame
		
		# Construct the action space.
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

		# RLVIC
		self.num_waypoints = num_waypoints
		# self.pos_set_list = np.zeros((2, 3))
		# self.quat_set_list = np.tile(identity_quat, (self.num_waypoints, 1))
		
		if stiffness is None:
			self.stiffness = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
		else:
			self.stiffness = np.ones(6) * stiffness
		
		if damping == 'auto':
			self.damping = 2 * np.sqrt(self.stiffness)
		else:
			# self.damping = np.ones(6) * damping
			self.damping = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
		
		self.null_space_damping = null_space_damping
		
		# Get the position, velocity, and actuator indices for the model.
		if controlled_joints is not None:
			self.sim_qpos_idx = get_qpos_indices(sim.model, controlled_joints)
			self.sim_qvel_idx = get_qvel_indices(sim.model, controlled_joints)
			self.sim_actuators_idx = get_actuator_indices(sim.model, controlled_joints)
			self.sim_joint_idx = get_joint_indices(sim.model, controlled_joints)
			
			self.self_qpos_idx = get_qpos_indices(self.model, controlled_joints)
			self.self_qvel_idx = get_qvel_indices(self.model, controlled_joints)
			self.self_actuators_idx = get_actuator_indices(self.model, controlled_joints)
			
			self.mujoco_config_kuka._connect(sim,
				joint_pos_addrs=get_qpos_indices(sim.model, controlled_joints),
				joint_vel_addrs=get_qvel_indices(sim.model, controlled_joints),
				joint_dyn_addrs=get_joint_indices(sim.model, controlled_joints)
											 )
		else:
			assert self.model.nv == self.model.nu, "if the number of degrees of freedom is different than the number of actuators you must specify the controlled_joints"
			self.sim_qpos_idx = range(self.model.nq)
			self.sim_qvel_idx = range(self.model.nv)
			self.sim_actuators_idx = range(self.model.nu)
			self.sim_joint_idx = range(self.model.nu)
			
			self.self_qpos_idx = range(self.model.nq)
			self.self_qvel_idx = range(self.model.nv)
			self.self_actuators_idx = range(self.model.nu)
			
			self.mujoco_config_kuka._connect(sim,
											 joint_pos_addrs=self.sim_qpos_idx,
											 joint_vel_addrs=self.sim_qvel_idx,
											 joint_dyn_addrs=self.sim_joint_idx
											 )

	def set_action(self, action):
		'''
			Set setpoint
		'''
		action = action * self.scale
		
		dx = action[0:3].astype(np.float64)
		dr = action[3:6].astype(np.float64)
		
		pos, mat = forwardKinSite(self.sim,
								  self.site_name,
								  recompute=False)
		quat = mat2Quat(mat)
		
		if self.in_ee_frame:
			dx = mat.dot(dx)
		
		self.pos_set = pos + dx
		self.quat_set = quatAdd(quat, dr)
		
	def get_state_cartersian_sapce(self, ):
		pos, mat = forwardKinSite(self.sim, self.site_name, recompute=False)
		_, _, xvel = forwardVelKinSite(self.sim, self.site_name, recompute=True)
		quat = mat2Quat(mat)
		
		return pos, quat, xvel
	
	def set_target(self, pos, quat):
		'''
			Set task space setpoint.
		'''
		self.pos_set = pos
		self.quat_set = quat
		
	def set_waypoints(self,
					  pos_list,
					  quat_list,
					  pos_optimal_point,
					  quat_optimal_point
					  ):
		'''
			Set task space setpoint list.
		'''
		self.pos_set_list = pos_list
		self.quat_set_list = quat_list
		self.num_waypoints = self.pos_set_list.shape[0]
		self.pos_optimal_point = pos_optimal_point
		self.quat_optimal_point = quat_optimal_point
		
	def get_torque_im(self, stiffness, weights):
		'''
			Update the impedance control set_point and compute the torque.
		'''
		# Compute the pose difference.
		pos, mat = forwardKinSite(self.sim, self.site_name, recompute=False)
		quat = mat2Quat(mat)
		
		jpos, jrot = forwardKinJacobianSite(self.sim, self.site_name, recompute=False)
		J = np.vstack((jpos[:, self.sim_qvel_idx], jrot[:, self.sim_qvel_idx]))
		cartesian_acc_des = np.zeros(6)
		stiffness_list = stiffness
		damping_list = 2 * np.sqrt(stiffness_list)
		
		# calculate difference with attractor points
		for i in range(self.num_waypoints):
			dx = self.pos_set_list[i] - pos
			# print('pos_list :::', self.pos_set_list[i])
			# print('quat_list :::', self.quat_set_list[i])
			dr = subQuat(self.quat_set_list[i], quat)
			dframe = np.concatenate((dx, dr))
			# print('Difference dframe :::', dframe)
			
			# Compute generalized forces from a virtual external force.
			cartesian_acc_des += weights[i] * stiffness_list[i] * dframe - \
								 weights[i] * damping_list[i] * J.dot(self.sim.data.qvel[self.sim_qvel_idx])
			
		# print("stiffness :::", stiffness_list[0])
		# print("damping :::", damping_list[0])
		# print("cartesian acc :::", cartesian_acc_des)
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
		
		return id_torque
	
	def get_torque(self):
		'''
			Update the impedance control setpoint and compute the torque.
		'''
		pos, mat = forwardKinSite(self.sim, self.site_name, recompute=False)
		quat = mat2Quat(mat)
		dx = self.pos_set - pos
		dr = subQuat(self.quat_set, quat)
		dframe = np.concatenate((dx, dr))
		# print('Difference dframe:::', dframe)
		# print("pos_set", self.pos_set)
		# print("quat_set", self.quat_set)
		
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
		
		return id_torque
	
	def get_torque_vic(self, stiffness, damping, weights):
		pos, mat = forwardKinSite(self.sim, self.site_name, recompute=True)
		quat = mat2Quat(mat)
		
		_, _, xvel = forwardVelKinSite(self.sim, self.site_name, recompute=True)
		
		jpos, jrot = forwardKinJacobianSite(self.sim, self.site_name, recompute=True)
		J = np.vstack((jpos[:, self.sim_qvel_idx], jrot[:, self.sim_qvel_idx]))
		cartesian_acc_des = np.zeros(6)
		
		stiffness_list = stiffness
		damping_list = damping
		
		# transform to new coordinate
		# current state: difference with optimal point
		state_pos = pos - self.pos_optimal_point
		state_quat = quat - self.quat_optimal_point
		# print("sim_state_pos :::", state_pos)
		# print("sim_state_quat :::", state_quat)
		# print("optimal_state_pos :::", self.pos_optimal_point)
		# print("optimal_state_quat :::", self.quat_optimal_point)
		
		# attractor point transform
		attractor_points_pos = self.pos_set_list - self.pos_optimal_point
		attractor_points_quat = self.quat_set_list - self.quat_optimal_point
		# print("attractor_points_pos :::", attractor_points_pos)
		
		omega_weights, beta_weights = self.get_non_linear_weight(state_pos=state_pos,
																 stiffness=stiffness_list,
																 weights=weights)
		
		# return equvilent stiffness and damping
		stiffness_eqv, damping_eqv = self.get_stiffness_damping(omega_weights=omega_weights,
																stiffness=stiffness,
																damping=damping)
		
		# print("weights :::", omega_weights)
		# sim_state = self.sim.get_state()
		# state_vel = J.dot(sim_state.qvel[:])
		state_vel = J.dot(self.sim.data.qvel[self.sim_qvel_idx])
		# print("sim_state_pos :::", state_pos)
		# print("sim_state_vel jacobian:::", state_vel[:3])
		# print("sim_state get :::", J.dot(self.sim.data.qvel[self.sim_qvel_idx]))
		# print("sim_state_vel_car :::", J.dot(sim_state.qvel[:]))
		# state_vel = np.zeros(6)
		# state_vel[:3] = xvel
		
		M_qq = self.mujoco_config_kuka.M(q=self.sim.data.qpos[self.sim_qpos_idx])
		
		# calculate energy function
		V = self.energy_function(
			state_pos=np.array(state_pos),
			state_vel=np.array(state_vel),
			stiffness=stiffness_list,
			weights=weights,
			beta_weights=beta_weights,
			M_qq=M_qq,
			J=J
		)
	
		# calculate difference with attractor points
		for i in range(self.num_waypoints):
			# print("i :::", i)
			# dx = state_pos - self.pos_set_list[i]
			dx = state_pos - attractor_points_pos[i]
			# print('pos_list :::', self.pos_set_list[i])
			# print('quat_list :::', self.quat_set_list[i])
			
			# dr = - subQuat(self.quat_set_list[i], state_quat)
			dr = - subQuat(attractor_points_quat[i], state_quat)
			# dr = quat2Vel(quat)
			
			dframe = np.concatenate((dx, dr))
			# print('Difference dframe :::', dframe)
			
			# Compute generalized forces from a virtual external force.
			cartesian_acc_des += - omega_weights[i] * stiffness_list[i] * dframe \
								 - omega_weights[i] * damping_list[i] * J.dot(self.sim.data.qvel[self.sim_qvel_idx])
		
		# print("stiffness :::", stiffness_list[0])
		# print("damping :::", damping_list[0])
		# print("cartesian acc :::", cartesian_acc_des)
		impedance_acc_des = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-6 * np.eye(6), cartesian_acc_des))
		
		# Add damping in the null space of Jacobian
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
		
		return id_torque, V, stiffness_eqv, damping_eqv, state_vel
	
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
	
	def get_non_linear_weight(self,
							  state_pos,
							  stiffness,
							  weights):
		"""
			get non-linear weights of DS
		"""
		self.alpha_weights = np.zeros_like(weights)
		self.beta_weights = np.zeros_like(weights)
		self.omega_weights = np.zeros_like(weights)
		self.omega_weights[0] = 1

		if self.num_waypoints > 1:
			for i in range(1, weights.shape[0]):
				if state_pos.dot(stiffness[i, :3] * np.eye(3)).dot(state_pos - self.pos_set_list[i]) >= 0:
					self.alpha_weights[i] = state_pos.dot(stiffness[i, :3] * np.eye(3)).dot(state_pos - self.pos_set_list[i])
				else:
					self.alpha_weights[i] = 0
				
				# print("alpha_weights ::::", self.alpha_weights[i])
				# print("original_weights ::::", weights[i])
				self.beta_weights[i] = np.exp(- 1 * (weights[i] / 4) * np.square(self.alpha_weights[i]))
				self.omega_weights[i] = self.alpha_weights[i] * self.beta_weights[i]
		
		# print(" omega weights :::", self.omega_weights)
		return self.omega_weights, self.beta_weights

	def energy_function(self,
						state_pos=None,
						state_vel=None,
						stiffness=None,
						weights=None,
						beta_weights=None,
						M_qq=None,
						J=None
						):
		"""
			Lyapunov function :::
		"""
		M_xx = np.linalg.inv(J.dot(np.linalg.inv(M_qq)).dot(J.T))
		# print("M_xx :::", M_xx.shape)
		# print("weights :::", weights)
		# print("beta :::", beta_weights)
		# print("state_pos :::", state_pos)
		# print("state_vel :::", state_vel)
		# print("Calculatin_1 :::", state_vel.dot(M_xx).dot(state_vel))
		# print("Calculatin_2 :::", (1/weights[1:] * (1 - beta_weights[1:])).sum())
		# print("Calculatin_3 :::", state_vel.dot(M_xx).dot(state_vel))
		
		state_vel[3:] = np.zeros(3)
		# state_vel[1] = 0
		V_1 = 1/2 * state_pos.dot(stiffness[0, :3] * np.eye(3)).dot(state_pos)
		V_2 = (1/weights[1:] * (1 - beta_weights[1:])).sum()
		V_3 = 1/2 * state_vel.dot(M_xx).dot(state_vel)
		V = V_1 + V_2 + V_3
		
		return V
	
	def get_stiffness_damping(self, omega_weights, stiffness, damping):
		"""
		"""
		self.stiffness_matrix = np.zeros(3) * np.eye(3)
		self.damping_matrix = np.zeros(3) * np.eye(3)
		for i in range(1, omega_weights.shape[0]):
			self.stiffness_matrix += omega_weights[i] * stiffness[i][:3] * np.eye(3)
			self.damping_matrix += omega_weights[i] * damping[i][:3] * np.eye(3)
			
		return self.stiffness_matrix, self.damping_matrix
		

register_controller(RLVIC, "RLVIC")
