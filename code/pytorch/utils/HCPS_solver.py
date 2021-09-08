import copy as cp
import os
import numpy as np
import torch
from ..methods import DDPG, TD3, SAC, AAC, HRLSAC, HRLACOP
from ..REPS.GPREPS import GPREPS
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
		self.L = self.args.num_simulated_average_episodes  # 每改变一次context参数，通过model获得artificial trajectory的次数
		self.max_episode_steps = self.args.max_episode_steps  # RL的最大步数
		self.eps_min = self.args.eps[0]
		self.eps_max = self.args.eps[1]
		self.reward_scale = self.args.reward_scale
		
		# definition from environment
		self.context_dim = self.env.context_dim
		self.latent_parameter_dim = self.env.latent_parameter_dim
		
		self.latent_parameter_low = self.env.latent_parameter_low
		self.latent_parameter_high = self.env.latent_parameter_high
		self.latent_parameter_initial = self.env.latent_parameter_initial
		
		# Set seeds
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)
		
		# ============================= lower-level RL policy =========================
		self.done = False
		self.safe = True
		state_dim = self.env.observation_space.shape[0]
		print("state_dim :::", state_dim)
		action_dim = self.env.action_space.shape[0]
		print("action_dim", action_dim)
		max_action = float(self.env.action_space.high[0])
		print('action_space_high', max_action)
		
		if 'TD3' == args.policy_name:
			policy = TD3.TD3(state_dim, action_dim, max_action)
		elif 'DDPG' == args.policy_name:
			policy = DDPG.DDPG(state_dim, action_dim, max_action)
		elif 'SAC' == args.policy_name:
			policy = SAC.SAC(args, state_dim, action_dim, max_action, self.env.action_space)
		elif 'AAC' == args.policy_name:
			policy = AAC.AAC(state_dim, action_dim, max_action)
		elif 'HRLSAC' == args.policy_name:
			policy = HRLSAC.HRLSAC(args, state_dim, action_dim, max_action)
		elif 'HRLACOP' == args.policy_name:
			policy = HRLACOP.HRLACOP(args, state_dim, action_dim, max_action, option_num=self.args.option_num)
		else:
			policy = DDPG.DDPG(state_dim, action_dim, max_action)
			print("Please give right control algorithm !!!")
		
		self.policy = policy
		
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
		
		# ============================= Model-based learning ===============================
		self.lower_replay_buffer = utils.ReplayBuffer(self.args.low_memory_size)
		self.higher_replay_buffer = utils.ReplayBuffer(self.args.high_memory_size)
		
		self.total_timesteps = 0
		self.episode_timesteps = 0
		self.episode_number = 0
		self.episode_reward = 0
		self.best_reward = 0
		
		# ============================== Data recording ===============================
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
		
		""" evaluation cps """
		self.evaluations_info_value = []
		self.evaluations_reward = []
		
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
		
		if self.args.load_policy:
			self.policy.load(self.file_name, self.args.load_path)
		
	def train_lower_once(self):
		"""
			train lower-level policy once
		"""
		if self.total_timesteps != 0 and self.total_timesteps > self.args.start_timesteps:
			# print("=================== Train Lower-level Policy Once ====================")
			self.policy.train(
				self.lower_replay_buffer, self.args.batch_size, self.args.discount_low,
				self.args.tau, self.args.policy_noise, self.args.noise_clip, self.args.policy_freq
				)
		
	def train_higher_once(self, k=0, mode="model_free"):
		"""
			train higher-level policy once
		"""
		print("=================== Train Higher-level Policy Once ====================")
		eps_k = self.eps_max - k * (self.eps_max - self.eps_min) / self.K
		if k > 10:
			if mode == "model_based":
				""" train reward model """
				self.gpreps.train_reward_model(N_samples=self.args.model_size, type='GP')
				
				""" Train context model """
				self.gpreps.train_context_model(N_samples=self.args.model_size, N_components=6)
				
				""" Sample context """
				z_list = self.gpreps.sample_context(self.M)
				z_list = z_list.reshape(-1, self.context_dim)
				w_list = self.gpreps.choose_action(z_list).reshape(-1, self.latent_parameter_dim)
				sample_reward = self.gpreps.generate_artificial_reward(z_list, w_list)
				
				for m in range(self.M):
					self.gpreps.store_simulated_data(z_list[m], w_list[m], sample_reward[m])
				
				self.gpreps.learn(
					training_type='Simulated',
					N_samples=self.args.num_training_samples,
					eps=eps_k
				)
			else:
				self.gpreps.learn(
					training_type='Realistic',
					N_samples=self.args.num_training_samples,
					eps=eps_k
				)
	
	def reset(self): 
		"""
			reset for each espisode training
		"""
		self.obs = self.env.reset()
		self.done = False
		self.safe = True
		self.episode_reward = 0
		self.episode_timesteps = 0
	
	def train(self):
		"""
			train policy
		"""
		for k in range(self.K):
			print("Policy Update K :::::", k)
			
			for i in range(self.N):
				# acquire context value z from environment
				z = self.env.get_context()
				z = z.reshape(-1, self.context_dim)
				
				w = self.gpreps.choose_action(np.array(z))
				self.env.controller.set_params(w[0])
				
				self.episode_number = 0.
				average_reward = 0.
				
				while self.episode_number < self.n:
					self.reset()
					
					while self.episode_timesteps < self.max_episode_steps:
						if self.total_timesteps % self.d == 0:
							self.train_lower_once()
						
						action = self.policy.select_action(np.array(self.obs))
						
						# action = np.random.normal(
						# 	action,
						# 	self.var,
						# 	size=self.env.action_space.shape[0]).clip(
						# 	self.env.action_space.low[0],
						# 	self.env.action_space.high[0]
						# )
						
						noise = np.random.normal(0, self.args.expl_noise, size=self.env.action_space.shape[0])
						
						if self.args.expl_noise != 0:
							action = (action + noise).clip(
								self.env.action_space.low, self.env.action_space.high
							)
							
						new_obs, reward, self.done, info = self.env.step(action)
						
						# self.var *= 0.9998
						done_bool = 0 if self.episode_timesteps + 1 == self.env._max_episode_steps else float(self.done)
						
						if self.render:
							self.env.render()
						
						# store data into lower-level replay buffer for lower-level RL training
						self.lower_replay_buffer.add((self.obs, new_obs, action, reward, done_bool, 0))
						
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
				
				self.gpreps.store_realistic_data(z[0], w[0], [np.around(average_reward/self.n * self.reward_scale, 4).copy()])
			
			""" train higher policy """
			self.train_higher_once(k=k, mode=self.args.training_type)
			
			if k % self.args.eval_policy_update == 0:
				self.eval_once()
		
		print("Final total time steps :::::::::::", self.total_timesteps)
		# np.save('memory_realistic.npy', np.array(self.gpreps.memory_realistic))
		np.save(self.log_dir + "/evaluations_reward", self.evaluations_reward)
		np.save(self.log_dir + "/evaluations_info_value", self.evaluations_info_value)
	
	def eval_once(self):
		
		print('::::::::::::::::::::::::::::::: evaluations ::::::::::::::::::::::::::::')
		self.evaluation_reward_step = np.zeros((self.args.eval_max_context_pairs, self.args.max_eval_episode, 1))
		self.evaluation_info_step = np.zeros((self.args.eval_max_context_pairs, self.args.max_eval_episode, len(self.info_keywords)))
		
		for i in range(self.args.eval_max_context_pairs):
			z = self.env.get_context()
			z = z.reshape(-1, self.context_dim)
			print("context z :::::", z)
			w = self.gpreps.choose_action(np.array(z))
			print('parameter w :::::', w)
			self.env.controller.set_params(w[0])
			
			for episode_number in range(self.args.max_eval_episode):
				self.reset()
				while self.episode_timesteps < self.max_episode_steps:
					action = self.policy.select_action(np.array(self.obs))
					action = action.clip(
						self.env.action_space.low, self.env.action_space.high
					)
					
					new_obs, reward, self.done, self.info = self.env.step(action)
					
					self.episode_reward += reward
					self.obs = new_obs
					self.episode_timesteps += 1
					if self.done or self.episode_timesteps == self.max_episode_steps - 1:
						print('RL episode', episode_number,
							  ', step', self.episode_timesteps,
							  'done?', self.done,
							  ', safe?', self.safe,
							  ', reward', np.round(self.episode_reward*self.reward_scale, 4)
							  )
						break
				
				self.evaluation_reward_step[i, episode_number, 0] = cp.deepcopy(np.round(self.episode_reward * self.reward_scale, 4))
				for j in range(len(self.info_keywords)):
					self.evaluation_info_step[i, episode_number, j] = cp.deepcopy(self.info[self.info_keywords[j]])
		
		self.evaluations_reward.append(cp.deepcopy(self.evaluation_reward_step))
		self.evaluations_info_value.append(cp.deepcopy(self.evaluation_info_step))
		
		np.save(self.log_dir + "/evaluations_reward", self.evaluations_reward)
		np.save(self.log_dir + "/evaluations_info_value", self.evaluations_info_value)
