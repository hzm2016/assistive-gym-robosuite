import os
import imageio
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py.builder import MujocoException
import mujoco_py
from mujoco_py import GlfwContext

from envs.mujoco.controllers import controller_registry
from envs.mujoco.envs.assets import kuka_asset_dir
from envs.mujoco.utils.quaternion import mat2Quat, subQuat


class KukaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	default_info = dict()
	
	def __init__(self,
				 controller,
				 controller_options,
				 reference_path=False,
				 model_path='full_kuka_no_collision_no_gravity.xml',
				 frame_skip=20,
				 save_video=True,
				 video_path=None,
				 time_limit=3.,
				 timestep=0.002,
				 random_model=False,
				 random_target=False,
				 quadratic_pos_cost=True,
				 quadratic_vel_cost=False
		):
		'''
			Constructs the file, sets the time limit and calls the constructor of
			the super class.
		'''
		self.random_model = random_model
		self.random_target = random_target
		self.quadratic_pos_cost = quadratic_pos_cost
		self.quadratic_vel_cost = quadratic_vel_cost
		self.reference_path = reference_path
		
		utils.EzPickle.__init__(self)
		
		full_path = os.path.join(kuka_asset_dir(), model_path)
		print("asset dir :::", kuka_asset_dir())
		print("full_path :::", full_path)
		self.time_limit = time_limit
		self.timer = 0
		
		# Parameters for the cost function
		self.state_des = np.zeros(14)
		self.Q_pos = np.diag([1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])
		self.Q_vel = np.diag([0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.])
		
		# Call the super class
		self.initialized = False
		mujoco_env.MujocoEnv.__init__(self, full_path, frame_skip)
		self.model.opt.timestep = timestep
		self.initialized = True
		
		# Create the desired controller.
		controller_cls = controller_registry[controller]
		self.controller = controller_cls(sim=self.sim, **controller_options)
		
		# Take the action space from the controller.
		self.action_space = self.controller.action_space
		self.last_action = None
		
		# save video or not
		self.save_video = save_video
		if self.save_video:
			GlfwContext(offscreen=True)
			self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=0)
			self.viewer.cam.trackbodyid = 0
			self.viewer.cam.distance = self.sim.model.stat.extent * 1.0
			self.viewer.cam.azimuth = 90
			self.viewer.cam.lookat[2] = 0.1
			self.viewer.cam.elevation = -15
			self.writer = imageio.get_writer(video_path, fps=10)
		else:
			self.viewer = mujoco_py.MjViewer(self.sim)
	
	def reset(self):
		self.sim.reset()
		ob = self.reset_model()
		target_obs, target_rot = self._get_target_obs()
		return ob, target_obs, target_rot
	
	def vic_reset(self, initial_offset=np.array([0., 0.035, 0.07]), pos_list=None, quat_list=None):
		
		self.sim.reset()
		ob = self.reset_model()
		
		# set initial state
		target_pos, target_rot = self._get_target_obs()
		
		# target position
		self.state_des = target_pos[1]
		quat_hole_top = mat2Quat(target_rot[2])
		pos_hole_top = target_pos[2]
		self.controller.set_target(pos_hole_top + initial_offset, quat_hole_top)
		
		# move to initial state
		while True:
			target_pos, target_rot = self._get_target_obs()
			quat_hole_top = mat2Quat(target_rot[2])
			target_distance = np.linalg.norm((target_pos[0] - self.controller.pos_set), ord=2)
			# print("target distance :::", target_distance)
			
			for _ in range(self.frame_skip):
				# torque = self.get_torque_im(stiffness, weights)
				torque = self._get_torque()
				self.sim.data.ctrl[:] = np.clip(torque, -100, 100)
				
				# add external force :::
				self.sim.data.qfrc_applied[:] = self._get_random_applied_force()
				
				# execute the current action
				self.sim.step()
			
			if target_distance < 0.0001:
				break
		
		# set attractor points :::
		self.set_waypoints(pos_list, quat_list,
						   pos_optimal_point=pos_list[0], quat_optimal_point=quat_list[0])

		final_pos = pos_list[0]
		# print("State Reset Done !!! final distance :::", target_distance)
		return ob, target_pos[2] + initial_offset, final_pos
	
	def set_target(self, pos, quat):
		"""
			set position and rotation
		"""
		self.controller.set_target(pos, quat)
		
	def set_waypoints(self, pos_list, quat_list, pos_optimal_point, quat_optimal_point):
		'''
			Set task space waypoint list.
		'''
		self.controller.set_waypoints(pos_list, quat_list, pos_optimal_point, quat_optimal_point)
	
	def viewer_setup(self):
		'''
			Overwrites the MujocoEnv method to make the camera point at the base.
		'''
		self.viewer.cam.trackbodyid = 0

	def step_im(self, stiffness, damping, weights, render=False):
		'''
			Simulate for `self.frame_skip` timesteps. Calls _update_action() once
			and then calls _get_torque() repeatedly to simulate a low-level
			controller.
			Optional argument render will render the intermediate frames for a smooth animation.
		'''
		
		# Hack to return an observation during the super class __init__ method.
		if not self.initialized:
			return self._get_obs(), 0, False, {}
		
		pos_0, quat_0, xvel_0 = self.controller.get_state_cartersian_sapce()
		
		# Simulate the low level controller.
		dt = self.sim.model.opt.timestep
		
		total_reward = 0
		action = np.zeros(6)
		
		total_reward_info = dict()
		target_pos, target_rot = self._get_target_obs()
		state = target_pos[0]
		# torque = np.zeros(7)
		
		try:
			for _ in range(self.frame_skip):
				# torque = self.controller.get_torque_im(stiffness, weights)
				torque, _, _, _, _ = self.controller.get_torque_vic(stiffness, damping, weights)
				# print("step im torque :::", torque)
				
				self.sim.data.ctrl[:] = np.clip(torque, -100, 100)
				
				# add external force :::
				self.sim.data.qfrc_applied[:] = self._get_random_applied_force()
				
				# execute the current action
				self.sim.step()
				
				if not np.all(np.isfinite(self.sim.data.qpos)):
					print("Warning: simulation step returned inf or nan.")

				if render:
					self.render()
				
			if self.save_video:
				# self.viewer.render(width=1024, height=1024)
				img = self.viewer.read_pixels(width=1024, height=1024, depth=False)
				frame = img[::-1, :, :]
				self.writer.append_data(frame)
			
			pos_1, quat_1, xvel_1 = self.controller.get_state_cartersian_sapce()
			action = pos_1 - pos_0
				
			# Get observation and check finished
			# (self.sim.data.time > self.time_limit) or
			done, safe = self.get_done(target_pos, final_range=0.004, safe_range=0.3)
			reward, reward_info = self.get_reward(pos_1, action, done, safe)
			
			total_reward += reward
			for k, v in reward_info.items():
				if 'reward' in k:
					total_reward_info[k] = total_reward_info.get(k, 0) + v * dt

			obs = self._get_obs()
			info = self._get_info()
			info.update(total_reward_info)
		except MujocoException as e:
			print(e)
			reward = 0
			obs = np.zeros_like(self.observation_space.low)
			done = False
			safe = False
			info = self.default_info
		
		return obs, total_reward, done, info, state, action, safe
	
	def step(self, action, render=False):
		'''
			Simulate for `self.frame_skip` timesteps. Calls _update_action() once
			and then calls _get_torque() repeatedly to simulate a low-level
			controller.
			Optional argument render will render the intermediate frames for a smooth animation.
		'''
		# Hack to return an observation during the super class __init__ method.
		if not self.initialized:
			return self._get_obs(), 0, False, {}
		
		# self._update_action(action)
		self.last_action = action
		
		# Simulate the low level controller.
		dt = self.sim.model.opt.timestep
		
		total_reward = 0
		total_reward_info = dict()
		target_pos, target_rot = self._get_target_obs()
		
		state = target_pos[0]
		try:
			for _ in range(self.frame_skip):
				torque = self._get_torque()
				# print('torque', torque)
				self.sim.data.ctrl[:] = np.clip(torque, -100, 100)
				
				# add simulated external force :::
				self.sim.data.qfrc_applied[:] = self._get_random_applied_force()
			
				# execute the current action
				self.sim.step()
				if not np.all(np.isfinite(self.sim.data.qpos)):
					print("Warning: simulation step returned inf or nan.")
				
				reward, reward_info = self._get_reward(state, action)
				total_reward += reward * dt
				for k, v in reward_info.items():
					if 'reward' in k:
						total_reward_info[k] = total_reward_info.get(k, 0) + v * dt
				if render:
					self.render()
			
			# Get observation and check finished
			# print("current time :::", self.sim.data.time)
			# print('time_out :::', (self.sim.data.time > self.time_limit))
			# (self.sim.data.time > self.time_limit) or
			done = self._get_done(target_pos)
			obs = self._get_obs()
			info = self._get_info()
			info.update(total_reward_info)
		except MujocoException as e:
			print(e)
			reward = 0
			obs = np.zeros_like(self.observation_space.low)
			done = True
			info = self.default_info
		
		return obs, total_reward, done, info, target_pos
	
	def get_reward(self, state, action, done, safe):
		'''
			Compute single step reward.
		'''
		# quadratic cost on the state error
		reward_info = dict()
		reward = 0.
		
		if done:
			reward += 10.0
		
		if safe is False:
			reward -= 30.0
		
		err = self.state_des - state
		if self.quadratic_pos_cost:
			reward_info['quadratic_pos_cost'] = -err.dot(self.Q_pos).dot(err)
			reward += reward_info['quadratic_pos_cost']
		
		# if self.quadratic_vel_cost:
		# 	reward_info['quadratic_vel_cost'] = -err.dot(self.Q_vel).dot(err)
		# 	reward += reward_info['quadratic_vel_cost']
		
		return reward, reward_info
	
	def get_done(self, pos, final_range=0.004, safe_range=0.3):
		'''
			Check the termination condition.
		'''
		done = False
		safe = True
		distance = np.linalg.norm((pos[0] - pos[1]), ord=2)
		
		# judge complete or not
		if distance < final_range:
			done = True
		
		# judge safe or not
		if distance > safe_range:
			safe = False
		
		return done, safe
	
	def _update_action(self, a):
		'''
			This function is called once per step.
		'''
		if self.reference_path:
			expert_action = self._expert_action()
			final_action = a * expert_action + expert_action
			self.controller.set_action(final_action)
		else:
			self.controller.set_action(a)
			
	def _expert_action(self,):
		"""
			Get the expert action by PD controller
		"""
		obs = self._get_obs()
		force = obs[:6]
		self.desired_force_moment = np.array([0.0, 0.0, -50, 0.0, 0.0, 0.0])
		self.kp = np.array([])
		self.kd = np.array([])
		
		force_error = self.desired_force_moment - force
		force_error *= np.array([-1, 1, 1, -1, 1, 1])

		""" Cal position of robot """
		if self.timer == 0:
			setPosition = self.kp * force_error[:3]
			self.former_force_error = force_error
		elif self.timer == 1:
			setPosition = self.kp * force_error[:3]
			self.last_setPosition = setPosition
			self.last_force_error = force_error
		else:
			setPosition = self.last_setPosition + self.kp * (force_error[:3] - self.last_force_error[:3]) + \
						  self.kd * (force_error[:3] - 2 * self.last_force_error[:3] + self.former_force_error[:3])
			self.last_setPosition = setPosition
			self.former_force_error = self.last_force_error
			self.last_force_error = force_error

		""" Cal orientation of robot """
		setEuler = self.kp[3:6] * force_error[3:6]

		movePosition = np.zeros(6)
		movePosition[:2] = np.clip(setPosition[:2], -0.5, 0.5)
		movePosition[2] = np.clip(setPosition[2], -0.5, -0.0)
		movePosition[3:6] = np.clip(setEuler, -0.2, 0.2)
		
		return movePosition
	
	def _get_torque(self):
		'''
			This function is called multiple times per step to simulate a
			low-level controller.
		'''
		return self.controller.get_torque()
	
	def _get_obs(self):
		'''
			Return the full state as the observation
		'''
		if self.random_target:
			return np.concatenate((self._get_state_obs(), self._get_target_obs()))
		else:
			return self._get_state_obs()
	
	def _get_state_obs(self):
		'''
			Return the observation given by the state.
		'''
		if not self.initialized:
			return np.zeros(14)
		
		return np.concatenate([self.sim.data.qpos[:], self.sim.data.qvel[:]])
	
	def _get_target_obs(self):
		raise NotImplementedError
	
	def _get_info(self):
		'''
			Get any additional info.
		'''
		q_err = self.state_des[:7] - self.sim.data.qpos
		v_err = self.state_des[7:] - self.sim.data.qvel
		dist = np.sqrt(q_err.dot(q_err))
		velocity = np.sqrt(v_err.dot(v_err))
		return {
			'distance': dist,
			'velocity': velocity
		}
	
	def _get_random_applied_force(self):
		return np.zeros(self.model.nv)
	
	def reset_model(self):
		'''
			Overwrites the MujocoEnv method to reset the robot state and return the observation.
		'''
		while (True):
			try:
				if self.random_model:
					self._reset_model_params()
				if self.random_target:
					self._reset_target()
				else:
					self._reset_fix_target()
				self._reset_state()  # Always reset the state after the target is reset.
				self.sim.forward()
			except MujocoException as e:
				print(e)
				continue
			break
		self.timer = 0
		return self._get_obs()
	
	def _reset_state(self):
		'''
			Reset the state of the model (i.e. the joint positions and velocities).
		'''
		qpos = 0.1 * self.np_random.uniform(low=self.model.jnt_range[:, 0],
											high=self.model.jnt_range[:, 1],
											size=self.model.nq)
		qvel = np.zeros(7)
		self.set_state(qpos, qvel)
	
	def _reset_target(self):
		'''
			Reset the goal parameters.
			Target pose for the base environment, but may change with subclasses.
		'''
		self.state_des[:7] = self.np_random.uniform(self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
	
	def _reset_model_params(self):
		'''
			TODO: implement this for domain randomization.
		'''
		raise NotImplementedError
