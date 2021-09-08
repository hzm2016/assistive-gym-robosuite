import copy as cp
import os
import torch

from code.pytorch.REPS.GPREPS import *
from results.result_analysis import *

from envs.abb.models import utils


class Solver(object):
	def __init__(self, args, env, project_path, result_path, info_keywords):
		self.args = args
		self.env = env
		self.file_name = ''
		self.project_path = project_path
		self.result_path = result_path
		self.render = self.args.render
		self.info_keywords = info_keywords
		
		# ############################# REPS Parameters ##########################
		self.K = self.args.num_policy_update  # 上层policy训练循环总数
		self.N = self.args.num_real_episodes  # 在上层policy的一个训练周期中，下层RL训练，改变context参数的次数
		self.n = self.args.num_average_episodes  # 下层RL训练，每改变一次context参数，执行RL的episode数
		self.d = self.args.policy_freq  # 下层RL每d个step更新一次网络参数
		self.M = self.args.num_simulated_episodes  # 在上层policy的一个训练周期中，通过model获得artificial trajectory的过程中，改变context参数的次数
		self.L = self.args.num_average_episodes  # 每改变一次context参数，通过model获得artificial trajectory的次数
		self.max_episode_steps = self.args.max_episode_steps  # RL的最大步数
		self.eps_min = self.args.eps[0]
		self.eps_max = self.args.eps[1]
		self.reward_scale = self.args.reward_scale
		
		# ############################# definition from env #######################
		self.context_dim = self.env.context_dim
		self.latent_parameter_dim = self.env.latent_parameter_dim
		
		self.latent_parameter_low = self.env.latent_parameter_low
		self.latent_parameter_high = self.env.latent_parameter_high
		self.latent_parameter_initial = self.env.latent_parameter_initial
		
		self.env = env
		self.done = False
		self.safe = True
		self.render = self.args.render
		
		self.file_name = ''
		self.project_path = project_path
		self.result_path = result_path
		
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)

		self.replay_buffer = utils.ReplayBuffer()
		self.replay_buffer_model = utils.ReplayBuffer(1e4)
		
		# =========================== High-level contextual policy =========================
		self.gpreps = GPREPS(
			self.context_dim,
			self.latent_parameter_dim,
			self.args.high_memory_size,
			self.latent_parameter_low,
			self.latent_parameter_high,
			self.latent_parameter_initial,
			self.eps_max
		)
		# ==================================================================================
		
		self.total_timesteps = 0
		self.episode_timesteps = 0
		self.episode_number = 0
		self.episode_reward = 0
		self.reward_scale = 0.01
		self.pre_num_steps = self.total_timesteps
		self.best_reward = 0.0
		self.timesteps_since_eval = 0
		
		""" training performance """
		self.training_reward = []
		self.training_time = []
		self.training_states = []
		self.training_im_actions = []
		
		""" evaluation performance """
		self.evaluations_time = []
		self.evaluations_actions = []
		self.evaluations_im_actions = []
		self.evaluations_states = []
		self.evaluations_options = []
		self.episode_reward_average_list = []
		self.successful_rate_list = []
		
		""" evaluation cps """
		self.evaluations_info_value = []
		self.evaluations_reward = []
		self.evaluations = []
		self.eval_episodes_states = []
		
		self.log_dir = '{}/{}/{}_{}_seed_{}'.format(
			self.result_path,
			self.args.log_path,
			self.args.policy_name,
			self.args.env_name,
			self.args.seed
		)
		
		print("---------------------------------------")
		print("Settings: %s" % self.log_dir)
		print("---------------------------------------")
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

	def reset(self):
		"""
			Random offset value :::
		"""
		self.obs = self.env.reset()
		self.done = False
		self.safe = True
		self.episode_reward = 0
		self.episode_timesteps = 0
	
	def train(self):
		"""
			train policy with REPS
		"""
		for k in range(self.K):
			print("Policy Update K :::::", k)
			
			for i in range(self.N):
				z = self.env.get_context()
				z = z.reshape(-1, self.context_dim)
				
				# obtain parameters from controller
				w = self.gpreps.choose_action(np.array(z))
				
				# interpret the impedance parameters
				# self.env.controller.set_params(w[0])
				self.env.set_controller_param(w[0])
				
				self.episode_number = 0.
				average_reward = 0.
				
				self.env.set_waypoints()
				
				while self.episode_number < self.n:
					""" environment reset """
					print("idx_episode :::", self.episode_number)
					self.reset()
					while self.episode_timesteps < self.max_episode_steps:
						
						# action = self.policy.select_action(np.array(self.obs))
						#
						# noise = np.random.normal(0, self.args.expl_noise, size=self.env.action_space.shape[0])
						#
						# if self.args.expl_noise != 0:
						# 	action = (action + noise).clip(
						# 		self.env.action_space.low, self.env.action_space.high
						# 	)
						
						action = np.zeros(6)
						new_obs, reward, self.done, info = self.env.step(action)
						done_bool = 0 if self.episode_timesteps + 1 == self.env._max_episode_steps else float(self.done)
						
						if self.render:
							self.env.render()
						
						# store data into lower-level replay buffer for lower-level RL training
						# self.lower_replay_buffer.add((self.obs, new_obs, action, reward, done_bool, 0))
						
						self.episode_reward += reward
						self.obs = new_obs
						self.episode_timesteps += 1
						self.total_timesteps += 1
						
						if self.done or self.episode_timesteps == self.max_episode_steps - 1:
							# print('RL episode', self.episode_number, ', step', self.episode_timesteps, 'done?', self.done,
							# 	  ', safe?', self.safe, ', reward', np.around(self.episode_reward * self.reward_scale, 4))
							self.episode_number += 1
							average_reward += self.episode_reward
							# higher replay buffer ::: higher contextual policy training
							# self.higher_replay_buffer.add((z[0], w, self.episode_reward))
							break
					print("episode_timesteps :::", self.episode_timesteps)
				
				self.gpreps.store_realistic_data(z[0], w, np.around(average_reward/self.n, 4))
			
			self.train_higher_once(k=k, mode="model-free")
			
			if k % self.args.eval_policy_update == 0:
				self.eval_once()
		
		print("Total time steps ::::", self.total_timesteps)
		np.save(self.log_dir + "/evaluations_reward", self.evaluations_reward)
		np.save(self.log_dir + "/evaluations_info_value", self.evaluations_info_value)
	
	def train_higher_once(self, k=0, mode="model-free"):
		"""
			train higher-level policy once
		"""
		print("=================== Train Higher-level Policy Once ====================")
		eps_k = self.eps_max - k * (self.eps_max - self.eps_min) / self.K
		if mode == "model-based":
			""" train reward model """
			self.gpreps.train_reward_model(N_samples=self.args.model_size, type='GP')
			
			""" Train dynamics model"""
			# self.gpreps.train_dynamics_model(State, Action)
			self.gpreps.train_context_model()
			for m in range(self.M):
				z = self.gpreps.sample_context()
				z = z.reshape(-1, self.context_dim)
				w = self.gpreps.choose_action(z)[0]
				
				# change reward model :::
				sample_reward = self.gpreps.generate_artificial_reward([np.array(z[0])], [w])
				self.gpreps.store_simulated_data(z[0], w, sample_reward)
			
			self.gpreps.learn(training_type='Simulated', eps=eps_k)
		else:
			self.gpreps.learn(
				training_type='Realistic',
				N_samples=self.args.num_training_samples,
				eps=eps_k
			)
	
	def eval_once(self):
		print(':::::::::::::::::::::: evaluations :::::::::::::::::::::')
		self.evaluation_reward_step = np.zeros(
			(self.args.eval_max_context_pairs, self.args.max_eval_episode, 1))
		self.evaluation_info_step = np.zeros(
			(self.args.eval_max_context_pairs, self.args.max_eval_episode, len(self.info_keywords)))
		
		for i in range(self.args.eval_max_context_pairs):
			z = self.env.get_context()
			z = z.reshape(-1, self.context_dim)
			print("context z :::::", z)
			w = self.gpreps.choose_action(np.array(z))
			print('parameter w :::::', w)
			self.env.controller.set_params(w[0])
			self.env.set_waypoints()
			for episode_number in range(self.args.max_eval_episode):
				self.reset()
				
				while self.episode_timesteps < self.max_episode_steps:
					# action = self.policy.select_action(np.array(self.obs))
					# action = action.clip(
					# 	self.env.action_space.low, self.env.action_space.high
					# )
					
					action = np.zeros(6)
					new_obs, reward, self.done, self.info = self.env.step(action)
					
					self.episode_reward += reward
					self.obs = new_obs
					self.episode_timesteps += 1
					if self.done or self.episode_timesteps == self.max_episode_steps - 1:
						print('RL episode', episode_number,
							  ', step', self.episode_timesteps,
							  'done?', self.done,
							  ', safe?', self.safe,
							  ', reward', np.round(self.episode_reward, 4)
							  )
						break
				
				self.evaluation_reward_step[i, episode_number, 0] = cp.deepcopy(
					np.round(self.episode_reward * self.reward_scale, 4))
				for j in range(len(self.info_keywords)):
					self.evaluation_info_step[i, episode_number, j] = cp.deepcopy(self.info[self.info_keywords[j]])
		
		self.evaluations_reward.append(cp.deepcopy(self.evaluation_reward_step))
		self.evaluations_info_value.append(cp.deepcopy(self.evaluation_info_step))
	
	def model_contextual_main(self):
		"""
			main function of GPRREPS for contextual policy training
		"""
		self.episode_reward_average_list = []
		self.successful_rate_list = []
		
		for k in range(self.K):
			print('Training Cycle...', k)
			successful_rate, episode_reward_average = self.train(random=True, type='GPREPS')
			
			# evaluate learning performance every cycle
			print('Successful Rate...', successful_rate)
			print("Average reward :::", episode_reward_average)
			self.episode_reward_average_list.append(cp.deepcopy(episode_reward_average))
			self.successful_rate_list.append(cp.deepcopy(successful_rate))
			
			if k > self.args.start_policy_update_idx:
				self.gpreps.train_reward_model(sample_number=200, type='GP')
				
				for m in range(self.M):
					z, _, _, _ = self.reset()
					z = self.get_context().reshape(-1, self.context_dim)
					w = self.gpreps.choose_action(z)[0]
					sample_reward = self.gpreps.generate_artifical_trajectory([np.array(z[0])], [w])
					self.gpreps.store_simulated_data(z[0], w, sample_reward)

			if k > self.args.start_policy_update_idx:
				print('****************** Start Training *****************')
				# eps_k = self.eps_min + k * (self.eps_max - self.eps_min) / self.K
				eps_k = self.eps_max - k * (self.eps_max - self.eps_min) / self.K
				self.gpreps.learn(training_type='Simulated', eps=eps_k)

			# if k % 100 == 0:
			# 	# Plot learning results
			# 	plot_single_reward(data=self.successful_rate_list, font_size=18,
			# 					   y_label_list=['Successful Rate'])
			# 	plot_single_reward(data=self.episode_reward_average_list, font_size=18,
			# 					   y_label_list=['Episode Reward Average'])
		
		# Save evaluation results
		np.save(self.result_path + "/successful_rate.npy", self.successful_rate_list)
		np.save(self.result_path + "/episode_reward.npy", self.episode_reward_average_list)
		
		# Plot learning results
		# plot_single_reward(data=self.successful_rate_list, font_size=18, y_label_list=['Episode Reward Average'])
		# plot_single_reward(data=self.episode_reward_average_list, font_size=18, y_label_list=['Episode Reward Average'])
		
		return self.episode_reward_average_list, self.successful_rate_list
