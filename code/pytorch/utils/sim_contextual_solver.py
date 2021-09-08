import copy as cp

import numpy as np

from ..multi_tasks_learning.GPREPS import *
from envs.mujoco.utils.quaternion import mat2Quat
from .plot_result import *

import time

from envs.abb.models import utils


class Solver(object):
	def __init__(self, args, env, project_path):
		
		self.args = args
		
		# #####################  Training Parameters  ######################
		self.K = self.args.num_policy_update  # 上层policy训练循环总数
		self.N = self.args.num_real_episodes  # 在上层policy的一个训练周期中，下层RL训练，改变context参数的次数
		self.n = 1  # 下层RL训练，每改变一次context参数，执行RL的episode数
		self.d = 1  # 下层RL每d个step更新一次网络参数
		self.M = self.args.num_simulated_episodes  # 在上层policy的一个训练周期中，通过model获得artificial trajectory的过程中，改变context参数的次数
		self.L = 5  # 每改变一次context参数，通过model获得artificial trajectory的次数
		
		# #####################  Policy Parameters  ######################
		self.MAX_EP_STEPS = self.args.max_episode_steps  # RL的最大步数
		self.context_dim = 6  # including initial state (x, y, z) and terminal state (x, y, z)
		self.num_waypoints = 3  # depends on the task to solve ::: predefined
		self.eps_max = 0.25
		self.eps_min = 0.10
		
		# Impedance parameters ::: stiffness damping is derived from a constant relation Kd = beta * sqrt(Kp)
		self.contextual_impedance_dim = 12
		self.contextual_impedance_lower_bound = np.array([0, 0.2, 0.2, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 2])
		self.contextual_impedance_upper_bound = np.array([0.0, 1.0, 1.0, 5.0, 5.0, 3.0, 5.0, 5.0, 2.0, 2.0, 2.0, 8])
		
		self.memory_dim = 500  # Context parameters
		# Give several initial expe
		# print("weights :::", weights)
		# print("stiffness pos :::", stiffness_pos)rt data ::: very important for policy search
		self.initial_a = np.array([1.0, 0.8, 0.1, 4, 4, 1, 3, 3, 1, 1, 1, 4])
		
		# attractor points
		self.pos_list = None
		self.quat_list = None
		
		# lower-level RL controller :::
		self.observation_dim = 12  # state
		self.action_dim = 6  # control command ::: action
		
		self.env = env
		self.done = False
		self.safe = True
		self.render = self.args.render
		
		self.file_name = ''
		self.project_path = project_path
		self.result_path = project_path + "results/runs/mujoco"
		
		self.evaluations = []
		self.eval_episodes_states = []
		
		# # Set seeds
		# torch.manual_seed(args.seed)
		# np.random.seed(args.seed)points
		
		# context parameter model
		self.replay_buffer = utils.ReplayBuffer()
		
		# lower-level memory
		self.replay_buffer_model = utils.ReplayBuffer(1e4)
		
		# Initialize GPREPS
		self.gpreps = GPREPS(self.context_dim, self.contextual_impedance_dim, self.memory_dim,
							 self.contextual_impedance_lower_bound, self.contextual_impedance_upper_bound,
							 self.initial_a, 0.25)
		
		# For model-based reps :::
		# self.r_model = R_MODEL(self.policy,
		# 					   self.env,
		# 					   self.context_dim,
		# 					   self.contextual_impedance_dim,
		# 					   self.observation_dim,
		# 					   self.action_dim,
		# 					   self.MAX_EP_STEPS)
		
		# self.s_model = S_MODEL()
		
		self.total_timesteps = 0
		self.episode_timesteps = 0
		self.episode_number = 0
		self.episode_reward = 0
		self.reward_scale = 0.001
		self.pre_num_steps = self.total_timesteps
		self.best_reward = 0.0
		self.timesteps_since_eval = 0
		
		""" training performance """
		self.training_reward = []
		self.training_time = []
		self.training_states = []
		self.training_im_actions = []
		
		""" evaluation performance """
		self.evaluations_reward = []
		self.evaluations_time = []
		self.evaluations_actions = []
		self.evaluations_im_actions = []
		self.evaluations_states = []
		self.evaluations_options = []
		
		self.episode_reward_average_list = []
		self.successful_rate_list = []
		
		self.episode_reward_average_list = []
		self.successful_rate_list = []
	
	def reset(self):
		"""
			Random offset value :::
		"""
		# initial point ::: add initial offset based on the top of hole
		initial_offset = np.zeros(3)
		initial_offset[:2] = np.random.uniform(-0.035, 0.035, size=2)
		initial_offset[2] = 0.07
		
		# final optimal point
		target_pos, target_rot = self.env._get_target_obs()
		quat_hole_top = mat2Quat(target_rot[2])
		quat_hole_base = mat2Quat(target_rot[1])
		
		# attractor points ::: could be constant
		self.pos_list = np.concatenate(([target_pos[1]], [target_pos[2]], [target_pos[2] - [0.0, 0.0, -0.05]]), axis=0)
		self.quat_list = np.concatenate(([quat_hole_base], [quat_hole_top], [quat_hole_top]), axis=0)
		# print("pos_list :::", self.pos_list)
		# print("quat_list :::", self.quat_list)
		
		self.obs, self.initial_state, self.target_state = \
			self.env.vic_reset(initial_offset=initial_offset, pos_list=self.pos_list, quat_list=self.quat_list)
		
		self.episode_reward = 0
		self.episode_timesteps = 0
		
		# successful episode number
		# self.episode_number = 0
		self.done = False
		self.safe = True
		self.state = self.initial_state
		
		# return context of one episode
		# self.context_state = np.concatenate((self.initial_state, self.target_state))
		self.context_state = self.get_context()
		return self.context_state, self.state, self.done, self.safe
	
	def get_context(self):
		# print('Get Context Parameter for one episode!!!')
		context = np.concatenate((self.initial_state, self.target_state))
		return context
	
	def train(self, w=None):
		"""
			execute reinforcement learning and store data for model training
		"""
		self.episode_number = 0
		self.episode_reward_average = 0
		for i in range(self.N):
			# print("Iteration ::::::::::::::::::::::", i)
			
			z, _, _, _ = self.reset()
			z = self.get_context().reshape(-1, self.context_dim)
			# print("context parameters :::", z)
			
			# choose impedance parameters ::: only stiffness and weights, damping is related to stiffness
			if w is None:
				w = self.gpreps.choose_action(z)[0]
			
			# print("Choose Impedance Parameters :::", w)
			
			# Extract Parameters for IMOGIC
			weights = np.array(w[:3])
			
			stiffness_pos = np.array([w[3:6], w[6:9], w[9:]])
			# print("weights :::", weights)
			# print("stiffness pos :::", stiffness_pos)
			
			stiffness_rotation = np.ones((3, 3))  # fixed in this experiment :::
			
			stiffness_list_test = np.concatenate((stiffness_pos, stiffness_rotation), axis=1)
			
			# stiffness_list = np.array([[4, 4, 0.1, 1, 1, 1],
			# 						   [0.5, 0.5, 8, 1, 1, 1]])
			# weights_list = np.array([0.9, 0.1])
			
			stiffness_list = stiffness_list_test
			beta = weights[0]
			damping_list = 2 * np.sqrt(stiffness_list)
			weights_list = weights
			weights_list[0] = 1
			
			# Start realistic simulation
			# while self.episode_number < self.n:
			# print("Episode Number :::", self.episode_number)
			# self.reset()
			while self.episode_timesteps < self.MAX_EP_STEPS:
				"""
					basic action from VIC
				"""
				obs, reward, self.done, info, self.state, action, self.safe = \
					self.env.step_im(stiffness=stiffness_list, damping=damping_list, weights=weights_list)
				
				new_obs = obs[:12]
				
				if self.render:
					self.env.render()
				# time.sleep(0.001)
				
				# store data into buffer ::: for RL training
				# self.replay_buffer.add((self.obs, new_obs, action, reward, self.done, 0))
				
				# here ::: self.obs and new_obs # for model training
				# self.replay_buffer_model.add((self.obs, new_obs, action))
				
				self.episode_reward += reward
				self.obs = new_obs
				self.episode_timesteps += 1
				
				if self.done or not self.safe or self.episode_timesteps == self.MAX_EP_STEPS - 1:
					print('RL episode :::', self.episode_number,
						  'step :::', self.episode_timesteps,
						  'done? :::', self.done,
						  'safe? :::', self.safe,
						  'episode reward :::', self.episode_reward)
					if self.done:
						# calculate successful rate
						self.episode_number += 1
					break
			
			# print("z :::", z)
			# print("w :::", w)
			# print("episode reward :::", self.episode_reward * 0.01)
			
			self.gpreps.store_realistic_data(z[0], w, self.episode_reward * self.reward_scale)
			self.episode_reward_average += self.episode_reward * self.reward_scale
		
		return self.episode_number / self.N, self.episode_reward_average / self.N
	
	def contextual_main(self):
		"""
			main function of GPRREPS for contextual policy training
		"""
		self.episode_reward_average_list = []
		self.successful_rate_list = []
		
		for k in range(self.K):
			print('Training Cycle...', k)
			successful_rate, episode_reward_average = self.train()
			
			# evaluate learning performance every cycle
			print('Successful Rate...', successful_rate)
			print("Average reward :::", episode_reward_average)
			self.episode_reward_average_list.append(cp.deepcopy(episode_reward_average))
			self.successful_rate_list.append(cp.deepcopy(successful_rate))
			
			# run RL and collect data
			# print('Running RL...')
			
			# train reward and context model
			# print('Training GMR models...')
			# self.r_model.train_reward_model(self.replay_buffer_model)
			
			# self.s_model.train_context_model()
			
			##################################
			# Predict Rewards and Store Data #
			##################################
			# print('Generate artificial trajectories !!!')
			# for j in range(self.M):
			# 	# Predict Rewards
			# 	R = 0.
			# 	Z = self.get_context()
			# 	W = self.gpreps.choose_action(Z)
			#
			# 	# Predict L Trajectories
			# 	for l in range(self.L):
			# 		R += self.r_model.trajectory(Z, W)
			# 	reward = R / self.L
			# 	print('Artificial running cycle', j + 1,
			# 		  'ra:', Z[0], 'rd:', Z[1],
			# 		  'reward:', reward)
			#
			# 	# Construct Artificial Dataset D
			# 	self.gpreps.store_simulated_data(Z, W, reward)
			
			# Sample and Update Policy
			# self.context_average = np.array([0.8, 0.8, 1.3, 0.8, 0.1, 1.29])
			#
			# print('memory :::', self.gpreps.memory_realistic)
			
			print('****************** Start Training *****************')
			eps_k = self.eps_min + k * (self.eps_max - self.eps_min) / self.K
			self.gpreps.learn(training_type='Realistic', eps=eps_k)
			
			# evaluation policy
			print('************ Training cycle finished **************')
		
		# Plot learning results
		plot_single_reward(data=self.successful_rate_list, font_size=18)
		plot_single_reward(self.episode_reward_average_list, font_size=18, y_label_list=['Episode Reward Average'])
		
		return self.episode_reward_average_list, self.successful_rate_list
	
	def model_contextual_main(self):
		"""
			main function of GPRREPS for contextual policy training
		"""
		self.episode_reward_average_list = []
		self.successful_rate_list = []
		
		for k in range(self.K):
			
			print('Training Cycle...', k)
			successful_rate, episode_reward_average = self.train()
			
			# evaluate learning performance every cycle
			print('Successful Rate...', successful_rate)
			print("Average reward :::", episode_reward_average)
			self.episode_reward_average_list.append(cp.deepcopy(episode_reward_average))
			self.successful_rate_list.append(cp.deepcopy(successful_rate))
			
			# run RL and collect data
			# print('Running RL...')
			
			# train reward and context model
			# print('Training GMR models...')
			# self.r_model.train_reward_model(self.replay_buffer_model)
			
			# self.s_model.train_context_model()
			
			##################################
			# Predict Rewards and Store Data #
			##################################
			# print('Generate artificial trajectories !!!')
			# for j in range(self.M):
			# 	# Predict Rewards
			# 	R = 0.
			# 	Z = self.get_context()
			# 	W = self.gpreps.choose_action(Z)
			#
			# 	# Predict L Trajectories
			# 	for l in range(self.L):
			# 		R += self.r_model.trajectory(Z, W)
			# 	reward = R / self.L
			# 	print('Artificial running cycle', j + 1,
			# 		  'ra:', Z[0], 'rd:', Z[1],
			# 		  'reward:', reward)
			#
			# 	# Construct Artificial Dataset D
			# 	self.gpreps.store_simulated_data(Z, W, reward)
			
			# Sample and Update Policy
			# self.context_average = np.array([0.8, 0.8, 1.3, 0.8, 0.1, 1.29])
			#
			# print('memory :::', self.gpreps.memory_realistic)
			
			if k > self.args.start_policy_update_idx:
				self.gpreps.train_reward_model(sample_number=100, type='GP')
				
				for m in range(self.M):
					z, _, _, _ = self.reset()
					z = self.get_context().reshape(-1, self.context_dim)
					w = self.gpreps.choose_action(z)[0]
					sample_reward = self.gpreps.generate_artifical_trajectory([np.array(z[0])], [w])
					self.gpreps.store_simulated_data(z[0], w, sample_reward)
			
			if k > self.args.start_policy_update_idx:
				print('****************** Start Training *****************')
				eps_k = self.eps_min + k * (self.eps_max - self.eps_min) / self.K
				self.gpreps.learn(training_type='Simulated', eps=eps_k)
			
			if k % 100 == 0:
				# Plot learning results
				plot_single_reward(data=self.successful_rate_list, font_size=18,
								   y_label_list=['Successful Rate'])
				plot_single_reward(data=self.episode_reward_average_list, font_size=18,
								   y_label_list=['Episode Reward Average'])
		
		# Plot learning results
		plot_single_reward(data=self.successful_rate_list, font_size=18)
		plot_single_reward(data=self.episode_reward_average_list, font_size=18, y_label_list=['Episode Reward Average'])
		
		return self.episode_reward_average_list, self.successful_rate_list
	
	def CGPUCB_main(self, num_rolluts=1000, beta=100):
		"""
			main function of GPUCB for contextual policy training
		"""
		# give initial parameters
		num_mesh_grid = 3
		mesh_grid_dist = np.array([0., 0.4, 0.4, 2.0, 2.0, 1.25, 2.0, 2.0, 1.0, 0.75, 0.75, 4.0])
		contextual_impedance_lower_bound = np.array([0.0, 0.2, 0.2, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 2])
		contextual_impedance_upper_bound = np.array([0.0, 1.4, 1.4, 7.0, 7.0, 4.25, 7.0, 7.0, 3.0, 2.75, 2.75, 12])
		
		para_samples = [np.array([0., 0., 0.])]
		for i in range(1, self.contextual_impedance_dim):
			sample_list = np.arange(contextual_impedance_lower_bound[i], contextual_impedance_upper_bound[i],
									mesh_grid_dist[i])
			para_samples.append(sample_list)
		
		# print("para_samples :::", para_samples)
		
		self.meshgrid = np.array(
			np.meshgrid(para_samples[0], para_samples[1], para_samples[2], para_samples[3], para_samples[4],
						para_samples[5],
						para_samples[6], para_samples[7], para_samples[8], para_samples[9], para_samples[10],
						para_samples[11]))
		
		self.sample_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
		
		self.mu = np.array([0. for _ in range(self.sample_grid.shape[0])])
		self.sigma = np.array([0.5 for _ in range(self.sample_grid.shape[0])])
		self.beta = beta
		# evaluation results
		
		self.realistic_sample_list = []
		self.episode_reward_average_list = []
		self.successful_rate_list = []
		
		gp = GaussianProcessRegressor()
		
		for k in range(num_rolluts):
			grid_idx = self.argmax_ucb(self.mu, self.sigma, self.beta)
			print("gird_idx", grid_idx)
			print("sample_gird", self.sample_grid)
			optimal_sample = self.sample_grid[grid_idx]
			print("optimal_sample", optimal_sample)
			successful_rate, episode_reward_average = self.train(w=optimal_sample)
			self.realistic_sample_list.append(optimal_sample)
			self.episode_reward_average_list.append(episode_reward_average)
			self.successful_rate_list.append(successful_rate)
			
			gp.fit(self.realistic_sample_list, self.episode_reward_average_list)
			self.mu, self.sigma = gp.predict(self.sample_grid, return_std=True)
		
		# Plot learning results
		plot_single_reward(data=self.successful_rate_list, font_size=18)
		plot_single_reward(self.episode_reward_average_list, font_size=18,
						   y_label_list=['Episode Reward Average'])
		
		return self.episode_reward_average_list, self.successful_rate_list
	
	def argmax_ucb(self, mu, sigma, beta):
		return np.argmax(mu + sigma * np.sqrt(beta))
	
	def eval_once(self):
		# self.pbar.update(self.total_timesteps - self.pre_num_steps)
		# self.pre_num_steps = self.total_timesteps
		# if self.timesteps_since_eval >= self.args.eval_freq:
		#     self.timesteps_since_eval %= self.args.eval_freq
		""" evaluate the policy for once """
		avg_reward, avg_time, eval_actions, eval_states, eval_im_actions, eval_options = \
			evaluate_assembly_policy(self.env, self.policy, self.args)
		self.evaluations_reward.append(cp.deepcopy(avg_reward))
		self.evaluations_time.append(cp.deepcopy(avg_time))
		self.evaluations_actions.append(cp.deepcopy(eval_actions))
		self.evaluations_states.append(cp.deepcopy(eval_states))
		self.evaluations_im_actions.append(cp.deepcopy(eval_im_actions))
		self.evaluations_options.append(cp.deepcopy(eval_options))
		print('evaluations_reward :::::::::::::::', self.evaluations_reward)
		print('evaluations_time :::::::::::::::::', self.evaluations_time)
		
		""" save test data numpy """
		np.save(self.log_dir + "/test_reward", self.evaluations_reward)
		np.save(self.log_dir + "/test_time", self.evaluations_time)
		np.save(self.log_dir + "/test_actions", self.evaluations_actions)
		np.save(self.log_dir + "/test_options", self.evaluations_options)
		np.save(self.log_dir + "/test_im_actions", self.evaluations_im_actions)
		np.save(self.log_dir + "/test_states", self.evaluations_states)
		
		utils.write_table(self.log_dir + "/test_reward", np.asarray(self.evaluations_reward))
		utils.write_table(self.log_dir + "/test_time", np.asarray(self.evaluations_time))
		
		if self.args.save_all_policy:
			self.policy.save(
				self.file_name + str(int(int(self.total_timesteps / self.args.eval_freq) * self.args.eval_freq)),
				directory=self.log_dir)
		else:
			self.policy.save(self.file_name, directory=self.log_dir)


# print('total_timesteps ::::::::::::::::::::::::::', self.total_timesteps)
# print('episode_reward :::::::::::::::::::::::::::', self.episode_reward)
# self.training_reward.append(cp.deepcopy(self.episode_reward))
# self.training_time.append(cp.deepcopy(self.episode_timesteps))
# self.training_states.append(cp.deepcopy(epi_states))
# self.training_im_actions.append(cp.deepcopy(epi_actions))
#
# np.save(self.log_dir + "/train_reward", self.training_reward)
# np.save(self.log_dir + "/train_time", self.training_time)
# np.save(self.log_dir + "/train_states", self.training_states)
# np.save(self.log_dir + "/train_im_actions", self.training_im_actions)
#
# utils.write_table(self.log_dir + "/train_reward", np.asarray(self.training_reward))
# utils.write_table(self.log_dir + "/train_time", np.asarray(self.training_time))


def evaluate_assembly_policy(env, policy, args):
	"""
		Runs policy for X episodes and returns average reward
	"""
	avg_reward = 0.
	eval_actions = []
	eval_im_actions = []
	eval_states = []
	eval_options = []
	start_time = time.time()
	for _ in range(args.num_eval_episodes):
		obs, state, done = env.reset()
		done = False
		episode_step = 0
		epi_actions = []
		epi_im_actions = []
		epi_options = []
		epi_states = []
		while not done and episode_step < args.max_episode_steps:
			if 'HRLACOP' in args.policy_name:
				action, option = policy.select_evaluate_action([np.array(obs)])
				epi_options.append(cp.deepcopy(option))
			else:
				action = policy.select_action(np.array(obs))
			
			epi_states.append(cp.deepcopy(state))
			obs, state, reward, done, _, execute_action = env.step(action)
			epi_actions.append(cp.deepcopy(action))
			epi_im_actions.append(cp.deepcopy(execute_action))
			avg_reward += reward
			episode_step += 1
		eval_states.append(cp.deepcopy(epi_states))
		eval_actions.append(cp.deepcopy(epi_actions))
		eval_im_actions.append(cp.deepcopy(epi_im_actions))
		eval_options.append(cp.deepcopy(epi_options))
	avg_time = (time.time() - start_time) / args.num_eval_episodes
	avg_reward /= args.eval_episodes
	return avg_reward, avg_time, eval_actions, eval_states, eval_im_actions, eval_options

# class GPREPS(object):
# 	def __init__(self, w_dim, z_dim, memory_dim, w_bound):
# 		# initialize parameters
# 		self.memory = []
# 		self.pointer = 0
# 		self.w_dim, self.z_dim, self.memory_dim, self.w_bound = w_dim, z_dim, memory_dim, w_bound
#
# 		# build actor
# 		self.a = np.ones((w_dim, 1), dtype=np.float32)
# 		self.A = np.ones((w_dim, z_dim), dtype=np.float32)
# 		self.COV = np.ones((w_dim, w_dim), dtype=np.float32)
#
# 	def choose_action(self, z):
# 		z = np.array([z])
# 		u = self.a + np.dot(self.A, z.transpose())
# 		u = u.transpose()[0]
# 		return np.random.multivariate_normal(mean=u, cov=self.COV, size=1)
#
# 	def learn(self, z_):
# 		eta, theta = argmin(self.memory, z_, self.z_dim)
#
# 		p = 0.
# 		P_ = []
# 		Z = []
# 		B = []
# 		for i in range(self.memory_dim):
# 			z, w, r = self.memory[i]
# 			z = np.array([z])
# 			w = np.array(w)
# 			r = np.array([r])
# 			p = np.exp((r - np.dot(z, theta)) / eta)
# 			z_ = np.c_[np.array([1.]), z]
# 			Z.append(z_[0])
# 			B.append(w[0])
# 			P_.append(p[0])
# 		p, B, Z = np.array(p), np.array(B), np.array(Z)
# 		P = P_ * np.eye(self.memory_dim)
#
# 		# calculate mean action
# 		target1 = np.linalg.inv(np.dot(np.dot(Z.transpose(), P), Z))
# 		# print(np.shape(P), np.shape(Z.transpose()), np.shape(B))
# 		target2 = np.dot(np.dot(Z.transpose(), P), B)
# 		target = np.dot(target1, target2).transpose()
# 		self.a = target[:, :1]
# 		self.A = target[:, 1:]
#
# 		# calculate the COV
# 		Err = 0
# 		for i in range(self.memory_dim):
# 			z, w, r = self.memory[i]
# 			z = np.array([z])
# 			w = np.array([w])
# 			err = w - self.a - np.dot(self.A, z.transpose())
# 			Err += np.dot(err, err.transpose()) * P_[i]
# 		self.COV = Err / np.sum(P_)
# 		# COV 减维到2维，否则维度是(1, 12, 12, 1)
# 		COV = np.zeros([12, 12])
# 		for i in range(12):
# 			for j in range(12):
# 				COV[i][j] = self.COV[0][i][j][0]
# 		print(np.shape(self.COV))
# 		print(np.shape(COV))
# 		self.COV = COV
# 		print('Contextual policy search upper level parameters updated')
#
# 	def store_data(self, z, w, r):
# 		transition = [z, w, [r]]
# 		if len(self.memory) == self.memory:
# 			index = self.pointer % self.memory_dim  # replace the old memory with new memory
# 			self.memory[index] = transition
# 		else:
# 			self.memory.append(transition)
# 		self.pointer += 1

# **** 程序结构 ****
# 主函数：solver.contextual_main()
# 运行RL: solver.train()
# 上层policy：class GPREPS
# 获得环境变量z：solver.get_context()
# model：class R_MODEL
# 训练model：R_MODEL.train_reward_model()
# 生成trajectory：R_MODEL.trajectory()
# w=u(z): class S_MODEL (未完成），目前是直接读取solver.get_context()

# **** 环境变量w设定 ****
# w = np.array([ra, rd)]
# ra：速度系数（0.6~1)，乘在action上面
# rd：深度系数（0.6~1)，乘在depth上面

# **** 待定参数： ****
# solver.init() 参数 K N d n M L w_boundary ......
# env
# safe or not: 安全边界力大小 (solver.train(), model.trajectory())
# trajectory初始位置obs 及其方差大小 (model.trajectory())
# trajectory reward function (model.trajectory())
