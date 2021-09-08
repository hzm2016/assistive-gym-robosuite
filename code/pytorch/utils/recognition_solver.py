import math
import os
import random
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle

from ..methods import DDPG, TD3, Average_TD3, SAC, HRLACOP
from envs.abb.models import utils


class Solver(object):
	def __init__(self, args, env, project_path):
		self.args = args
		self.env = env
		
		self.file_name = ''
		self.project_path = project_path
		self.result_path = project_path + "results/runs/mujoco"
		
		self.evaluations = []
		self.estimate_Q_vals = []
		self.true_Q_vals = []
		self.eval_episodes_states = []
		
		# Set seeds
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)
		
		state_dim = env.observation_space.shape[0]
		print('state_dim', state_dim)
		action_dim = env.action_space.shape[0]
		print('action_dim', action_dim)
		print(env.action_space.high)
		max_action = float(env.action_space.high[0])
		
		# Initialize policy
		if 'ATD3' == args.policy_name:
			policy = ATD3.ATD3(state_dim, action_dim, max_action)
		elif 'ATD3_RNN' == args.policy_name:
			policy = ATD3_RNN.ATD3_RNN(state_dim, action_dim, max_action)
		elif 'DDPG_RNN' == args.policy_name:
			policy = DDPG_RNN.DDPG_RNN(state_dim, action_dim, max_action)
		elif 'TD3_RNN' == args.policy_name:
			policy = TD3_RNN.TD3_RNN(state_dim, action_dim, max_action)
		elif 'Average_TD3' == args.policy_name:
			policy = Average_TD3.Average_TD3(state_dim, action_dim, max_action, num_steps=args.average_steps)
		elif 'DDPG' == args.policy_name:
			policy = DDPG.DDPG(state_dim, action_dim, max_action)
		elif 'SAC' == args.policy_name:
			policy = SAC.SAC(state_dim, action_dim, max_action, self.env.action_space)
		elif 'HRLACOP' == args.policy_name:
			policy = HRLACOP.HRLACOP(args, state_dim, action_dim, max_action, option_num=self.args.option_num)
		else:
			policy = TD3.TD3(state_dim, action_dim, max_action)
		
		if 'Average' in self.args.policy_name:
			self.log_dir = '{}/{}/{}_{}_{}_seed_{}'.format(self.result_path,
														   self.args.log_path,
														   self.args.policy_name,
														   self.args.average_steps,
														   self.args.env_name,
														   self.args.seed)
		elif 'HRLACOP' in self.args.policy_name:
			self.log_dir = '{}/{}/{}_{}_{}_seed_{}'.format(self.result_path,
														   self.args.log_path,
														   self.args.policy_name,
														   self.args.option_num,
														   self.args.env_name,
														   self.args.seed)
			
			self.log_transfer_dir = '{}/{}_transfer/{}_{}_{}_seed_{}'.format(self.result_path,
																			 self.args.log_path,
																			 self.args.policy_name,
																			 self.args.option_num,
																			 self.args.env_name,
																			 self.args.seed)
		else:
			self.log_dir = '{}/{}/{}_{}_seed_{}'.format(self.result_path,
														self.args.log_path,
														self.args.policy_name,
														self.args.env_name,
														self.args.seed)
			
			self.log_transfer_dir = '{}/{}_transfer/{}_{}_seed_{}'.format(self.result_path,
																		  self.args.log_path,
																		  self.args.policy_name,
																		  self.args.env_name,
																		  self.args.seed)
		
		print("---------------------------------------")
		print("Settings: %s" % self.log_dir)
		print("---------------------------------------")
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		
		print("---------------------------------------")
		print("Settings: %s" % self.log_transfer_dir)
		print("---------------------------------------")
		if not os.path.exists(self.log_transfer_dir):
			os.makedirs(self.log_transfer_dir)
		
		self.policy = policy
		self.replay_buffer = utils.ReplayBuffer()
		
		# data efficient hrl
		self.replay_buffer_high = utils.ReplayBufferHighLevel()
		self.replay_buffer_low = utils.ReplayBufferOption()
		
		self.total_timesteps = 0
		self.pre_num_steps = self.total_timesteps
		self.timesteps_since_eval = 0
		self.timesteps_calc_Q_vale = 0
		self.best_reward = 0.0
	
	def train_once(self):
		if self.total_timesteps != 0:
			if self.args.evaluate_Q_value:
				self.writer_train.add_scalar('ave_reward', self.episode_reward, self.total_timesteps)
			
			if 'HRLACOP' in self.args.policy_name:
				self.policy.train(
					self.replay_buffer_low,
					self.replay_buffer_high,
					batch_size_lower=self.args.batch_size,
					batch_size_higher=self.args.option_batch_size,
					discount_higher=self.args.discount_high,
					discount_lower=self.args.discount_low,
					tau=self.args.tau,
					policy_freq=self.args.policy_freq
				)
			else:
				self.policy.train(self.replay_buffer,
								  self.args.batch_size,
								  self.args.discount_low,
								  self.args.tau,
								  self.args.policy_noise,
								  self.args.noise_clip,
								  self.args.policy_freq)
	
	def eval_once(self):
		self.pbar.update(self.total_timesteps - self.pre_num_steps)
		self.pre_num_steps = self.total_timesteps
		
		# Evaluate episode
		if self.timesteps_since_eval >= self.args.eval_freq:
			self.timesteps_since_eval %= self.args.eval_freq
			
			# evaluate the policy for once
			avg_reward, eval_states, avg_episode_steps = evaluate_policy(self.env, self.policy, self.args)
			self.evaluations.append(avg_reward)
			self.eval_episodes_states.append(eval_states)
			self.writer_test.add_scalar('ave_reward', avg_reward, self.total_timesteps)
			
			if self.best_reward < avg_reward:
				self.best_reward = avg_reward
				print("Best reward! Total T: %d Episode T: %d Reward: %f" %
					  (self.total_timesteps, self.episode_timesteps, avg_reward))
				self.policy.save(self.file_name, directory=self.log_dir)
				np.save(self.log_dir + "/test_accuracy", self.evaluations)
				utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
	
	def reset(self):
		self.obs = self.env.reset()
		self.high_obs = self.obs
		self.obs_vec = np.dot(np.ones((self.args.seq_len, 1)), self.obs.reshape((1, -1)))
		self.episode_reward = 0
		self.episode_timesteps = 0
	
	def train(self):
		avg_reward, eval_states, _ = evaluate_policy(self.env, self.policy, self.args)
		self.evaluations = [avg_reward]
		self.eval_episodes_states.append(eval_states)
		
		# TesnorboardX
		if self.args.evaluate_Q_value:
			self.writer_train = SummaryWriter(logdir=self.log_dir + '_train')
		
		self.writer_test = SummaryWriter(logdir=self.log_dir)
		self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
		
		if self.args.load_policy:
			self.policy.load(self.file_name + str(self.args.load_policy_idx), self.log_dir)
		
		done = False
		safe_or_not = True
		self.cumulative_reward = 0.
		self.steps_done = 0
		option_data = []
		self.reset()
		
		while self.total_timesteps < self.args.max_timesteps:
			self.train_once()
			if done or not safe_or_not or self.episode_timesteps + 1 > self.args.max_episode_steps:
				print('safe_or_not', safe_or_not)
				print('done', done)
				print('total_timesteps', self.total_timesteps)
				print('episode_reward', self.episode_reward)
				self.eval_once()
				self.reset()
				done = False
				safe_or_not = True
			
			# Select action randomly or according to policy
			if self.total_timesteps < self.args.start_timesteps:
				action = self.env.action_space.sample()
				p = 1
				self.option = np.random.randint(self.args.option_num)
				self.next_option = np.random.randint(self.args.option_num)
			else:
				if 'RNN' in self.args.policy_name:
					action = self.policy.select_action(np.array(self.obs_vec))
				elif 'SAC' in self.args.policy_name:
					action = self.policy.select_action(np.array(self.obs), eval=False)
				elif 'HRLACOP' == self.args.policy_name:
					EPS_START = 0.9
					EPS_END = 0.05
					EPS_DECAY = self.args.max_timesteps
					# change option and calculate reward
					if (self.total_timesteps > self.args.start_timesteps) and (
							self.total_timesteps % self.args.option_change == 0):
						# print(self.total_timesteps)
						# change option every K steps ::::::
						sample = random.random()
						eps_threshold = EPS_END + (EPS_START - EPS_END) * \
										math.exp(-1. * self.total_timesteps / EPS_DECAY)
						self.steps_done += 1
						self.next_high_obs = self.obs
						
						if sample > eps_threshold:
							# option, _, _ = self.policy.softmax_option_target([np.array(self.obs)])
							# self.next_option = option.cpu().data.numpy().flatten()[0]
							action, self.option = self.policy.select_action(np.array(self.obs),
																			self.option,
																			change_option=True)
						else:
							self.option = np.random.randint(self.args.option_num)
						
						self.replay_buffer_high.add(
							(self.high_obs, self.next_high_obs, self.option, self.next_option, self.cumulative_reward))
						self.high_obs = self.next_high_obs
						
						self.auxiliary_reward = self.cumulative_reward / self.args.option_change
						option_data = np.array(option_data)
						option_data[:, -2] = self.auxiliary_reward
						for i in range(len(option_data)):
							self.replay_buffer_low.add(option_data[i])
						option_data = []
						
						self.cumulative_reward = 0.
					else:
						action, self.option = self.policy.select_action(np.array(self.obs),
																		self.option,
																		change_option=False)
				else:
					action = self.policy.select_action(np.array(self.obs))
				
				if self.args.expl_noise != 0:
					action = (action + np.random.normal(0, self.args.expl_noise,
														size=self.env.action_space.shape[0])).clip(
						self.env.action_space.low[0], self.env.action_space.high[0])
			
			new_obs, reward, done, _ = self.env.step(action)
			self.env.render()
			
			self.cumulative_reward += reward
			self.episode_reward += reward
			auxiliary_reward = 0.
			
			done_bool = 0 if self.episode_timesteps + 1 == self.args.max_episode_steps else float(done)
			
			if 'RNN' in self.args.policy_name:
				# Store data in replay buffer
				new_obs_vec = utils.fifo_data(np.copy(self.obs_vec), new_obs)
				self.replay_buffer.add((np.copy(self.obs_vec), new_obs_vec, action, reward, done_bool))
				self.obs_vec = utils.fifo_data(self.obs_vec, new_obs)
			elif 'HRLACOP' == self.args.policy_name:
				if self.total_timesteps <= self.args.start_timesteps:
					self.replay_buffer_low.add(
						(self.obs, new_obs, action, self.option, self.next_option, reward, auxiliary_reward, done_bool))
				else:
					option_data.append(
						(self.obs, new_obs, action, self.option, self.next_option, reward, auxiliary_reward, done_bool))
			else:
				self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool, p))
			
			self.obs = new_obs
			self.episode_timesteps += 1
			self.total_timesteps += 1
			self.timesteps_since_eval += 1
			self.timesteps_calc_Q_vale += 1
		
		avg_reward, eval_states, avg_episode_steps = evaluate_policy(self.env, self.policy, self.args)
		self.evaluations.append(avg_reward)
		self.eval_episodes_states.append(eval_states)
		
		if self.best_reward < avg_reward:
			self.best_reward = avg_reward
			print("Best reward! Total T: %d Episode T: %d Reward: %f" %
				  (self.total_timesteps, self.episode_timesteps, avg_reward))
			self.policy.save(self.file_name, directory=self.log_dir)
		
		if self.args.save_all_policy:
			self.policy.save(self.file_name + str(int(self.args.max_timesteps)), directory=self.log_dir)
		
		if self.args.load_policy:
			np.save(self.log_transfer_dir + "/test_accuracy", self.evaluations)
			utils.write_table(self.log_transfer_dir + "/test_accuracy", np.asarray(self.evaluations))
		else:
			np.save(self.log_dir + "/test_accuracy", self.evaluations)
			np.save(self.log_dir + "/episode_states", self.eval_episodes_states)
			utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
		
		# # save the replay buffer
		# if self.args.save_data:
		#     self.replay_buffer_low.save_buffer(self.log_dir + "/buffer_data")
		
		self.env.reset()
	
	def eval_only(self):
		self.evaluations = [evaluate_policy(self.env, self.policy, self.args)]
		
		# add tensorboard data
		self.writer_test = SummaryWriter(logdir=self.log_dir + '_test')
		self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
		
		if self.args.load_policy:
			self.policy.load(self.file_name + str(self.args.load_policy_idx), self.log_dir)
		
		done = False
		safe_or_not = True
		self.cumulative_reward = 0.
		self.steps_done = 0
		self.reset()
		while self.total_timesteps < self.args.eval_max_timesteps:
			if done or not safe_or_not or self.episode_timesteps + 1 > self.args.max_episode_steps:
				print('safe_or_not', safe_or_not)
				print('done', done)
				print('total_timesteps', self.total_timesteps)
				print('episode_reward', self.episode_reward)
				self.eval_once()
				self.reset()
				done = False
				safe_or_not = True
			
			# Select action randomly or according to policy
			if 'RNN' in self.args.policy_name:
				action = self.policy.select_action(np.array(self.obs_vec))
			elif 'SAC' in self.args.policy_name:
				action = self.policy.select_action(np.array(self.obs), eval=False)
			elif 'HRLACOP' == self.args.policy_name:
				if (self.total_timesteps % self.args.option_change == 0):
					self.steps_done += 1
					self.next_high_obs = self.obs
					action, self.option = self.policy.select_action(np.array(self.obs),
																	self.option,
																	change_option=True)
				
				else:
					action, self.option = self.policy.select_action(np.array(self.obs),
																	self.option,
																	change_option=False)
			else:
				action = self.policy.select_action(np.array(self.obs))
			
			new_obs, reward, done, _ = self.env.step(action)
			
			self.cumulative_reward += reward
			self.episode_reward += reward
			
			done_bool = 0 if self.episode_timesteps + 1 == self.args.max_episode_steps else float(done)
			
			self.obs = new_obs
			self.episode_timesteps += 1
			self.total_timesteps += 1
			self.timesteps_since_eval += 1
			self.timesteps_calc_Q_vale += 1
		
		avg_reward = evaluate_policy(self.env, self.policy, self.args)
		self.evaluations.append(avg_reward)
		print('evaluations', self.evaluations)
		
		if self.best_reward < avg_reward:
			self.best_reward = avg_reward
			print("Best reward! Total T: %d Episode T: %d Reward: %f" %
				  (self.total_timesteps, self.episode_timesteps, avg_reward))
			self.policy.save(self.file_name, directory=self.log_dir)
		
		if self.args.save_all_policy:
			self.policy.save(self.file_name + str(int(self.args.max_timesteps)), directory=self.log_dir)
		
		if self.args.load_policy:
			np.save(self.log_transfer_dir + "/test_accuracy", self.evaluations)
			utils.write_table(self.log_transfer_dir + "/test_accuracy", np.asarray(self.evaluations))
		else:
			np.save(self.log_dir + "/test_accuracy", self.evaluations)
			utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
		
		self.env.reset()
	
	def auto(self):
		for i in range(1):
			print('episode', i + 1, ':')
			self.env.reset()
			obs = random_start(self.env)
			print("obs :::", obs)
			step = 0
			temporary_buffer = []
			safe = True

			# phase 1: approach
			while obs[20] >= 0.02:
				action = np.array([0, 0, 0.02, 0, 0, 0])
				obs, reward, done, _ = self.env.step(action)
				temporary_buffer.append(np.hstack([1, obs]))
				self.env.render()
				print('phase 1, step', step, ':', obs[:6])
				step += 1
				# safe?
				if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
					safe = False
			
			# phase 2: contact
			while obs[20] >= 0.005:
				action = np.array([0, 0, 0.01, 0, 0, 0])
				obs, reward, done, _ = self.env.step(action)
				temporary_buffer.append(np.hstack([2, obs]))
				self.env.render()
				print('phase 2, step', step, ':', obs[:6])
				step += 1
				# safe?
				if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
					safe = False
			
			# phase 3: fit
			while np.abs(obs[-3]) >= 0.001 or np.abs(obs[-2]) >= 0.001:
				e = obs[-3:-1]
				le = (e[0] ** 2 + e[1] ** 2) ** 0.5
				action = 0.02 * e / le
				action_z = np.random.normal(0, 0.01)
				action = np.hstack([action, np.array([action_z, 0, 0, 0])])
				if le <= 0.01:
					obs, reward, done, _ = self.env.step(action / 2)
				else:
					obs, reward, done, _ = self.env.step(action)
				temporary_buffer.append(np.hstack([3, obs]))
				print('phase 3, step', step, ':', obs[:6])
				step += 1
				# safe?
				if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
					safe = False
				self.env.render()
			
			# phase 4: align
			while np.abs(obs[9]) >= 0.001 or np.abs(obs[10]) >= 0.001:
				w = 0.2
				a1 = 0
				a2 = 0
				if np.abs(obs[9]) >= 0.001:
					if obs[9] >= 0:
						a1 = w
					else:
						a1 = -w
				if np.abs(obs[10]) >= 0.001:
					if obs[10] >= 0:
						a2 = w
					else:
						a2 = -w
				action = np.array([0, 0, 0, a1, a2, 0])
				obs, reward, done, _ = self.env.step(action / 5)
				temporary_buffer.append(np.hstack([4, obs]))
				self.env.render()
				print('phase 4, step', step, ':', obs[:6])
				step += 1
				# safe?
				if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
					safe = False
			
			# phase 5: insertion
			while obs[8] >= 0:
				action = np.array([0, 0, 0.01, 0, 0, 0])
				obs, reward, done, _ = self.env.step(action)
				temporary_buffer.append(np.hstack([5, obs]))
				self.env.render()
				print('phase 5, step', step, ':', obs[:6])
				step += 1
				# safe?
				if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
					safe = False
			
			import matplotlib.pyplot as plt
			# # plot the trajectory of force
			fx = []
			fy = []
			fz = []
			s = []
			for t in range(step):
			    fx.append(temporary_buffer[t][0])
			    fy.append(temporary_buffer[t][1])
			    fz.append(temporary_buffer[t][2])
			    s .append(t+1)
			plt.plot(s, fx)
			plt.plot(s, fy)
			plt.plot(s, fz)
			plt.show()
			
			# if success, store the data into the memory
			if safe:
				for t in range(len(temporary_buffer)):
					self.replay_buffer.add(temporary_buffer[t][6:])
			print('safe?', safe, 'total step number:', step)
			print('buffer lenth:', len(self.replay_buffer.storage))
			print('*********************** episode finished ************************')
			if len(self.replay_buffer.storage) == 1000000:
				break
		
		# save replay buffer for furthor use
		f = open('replay_buffer_test.pckl', 'wb')
		pickle.dump(self.replay_buffer.storage, f)
		f.close()
	
	def test1(self):
		obs = self.env.reset()
		for i in range(200):
			action = np.array([0.01, 0, 0, 0, 0, 0])
			obs, reward, done, _ = self.env.step(action)
			self.env.render()
			print(obs)
		for i in range(400):
			action = np.array([-0.01, 0, 0, 0, 0, 0])
			obs, reward, done, _ = self.env.step(action)
			self.env.render()
			print(obs)
		for i in range(200):
			action = np.array([0.01, 0, 0, 0, 0, 0])
			obs, reward, done, _ = self.env.step(action)
			self.env.render()
			print(obs)
	
	def test2(self):
		obs = self.env.reset()
		for i in range(200):
			action = np.array([0.01, 0, 0, 0, 0, 0])
			obs, reward, done, _ = self.env.step(action)
			self.env.render()
			print(obs[-2])
		while obs[-3] >= 0.001 or obs[-3] <= -0.001:
			print(obs[12])
			if obs[-3] <= 0:
				action = np.array([-0.01, 0, 0, 0, 0, 0])
			else:
				action = np.array([0.01, 0, 0, 0, 0, 0])
			print(action)
			obs, reward, done, _ = self.env.step(action)
			self.env.render()
			print(obs)


def evaluate_policy(env, policy, args, eval_episodes=1):
	""" Runs policy for X episodes and returns average reward """
	avg_reward = 0.
	avg_episode_steps = 0
	save_state = True
	if save_state == True:
		evaluate_episode_states = []
	for _ in range(eval_episodes):
		print('eval_episodes', eval_episodes)
		obs = env.reset()
		# obs, state, done = env.reset()
		done = False
		safe_or_not = True
		eval_episodes_steps = 0
		episode_states = []
		while not done and safe_or_not and eval_episodes_steps < args.max_episode_steps:
			if 'HRLACOP' in args.policy_name:
				# without any exploration
				action = policy.select_evaluate_action([np.array(obs)])
			else:
				action = policy.select_action(np.array(obs))
			
			# obs, _, reward, done, safe_or_not = env.step(action)
			obs, reward, done, _ = env.step(action)
			episode_states.append(obs)
			avg_reward += reward
			avg_episode_steps += 1
			eval_episodes_steps += 1
		if save_state == True:
			evaluate_episode_states.append(episode_states)
	avg_reward /= eval_episodes
	avg_episode_steps /= eval_episodes
	print('eval_avg_reward', avg_reward)
	return avg_reward, evaluate_episode_states, avg_episode_steps


def cal_true_value(env, policy, replay_buffer, args, eval_episodes=1000):
	true_Q_val_vec = []
	init_state_vec, _, _, _, _ = replay_buffer.sample(eval_episodes)
	for i in range(eval_episodes):
		env.reset()
		if 'RNN' in args.policy_name:
			obs, obs_error = env.set_robot(init_state_vec[i][-1])
			obs_vec = np.copy(init_state_vec[i])
			obs_vec[-1] = np.copy(obs)
		else:
			obs, obs_error = env.set_robot(init_state_vec[i])
		true_Q_value = 0.
		if obs_error > 1e-3:
			print('Error of resetting robot: {},\n input obs: {},\n output obs: {}'.format(
				obs_error, init_state_vec[i], obs))
			continue
		done = False
		dis_gamma = 1.
		while not done:
			if 'RNN' in args.policy_name:
				action = policy.select_action(np.array(obs_vec))
			else:
				action = policy.select_action(np.array(obs))
			
			if 'IM' in args.policy_name:
				action_im = np.copy(action)
				action = utils.calc_torque_from_impedance(action_im, np.asarray(obs)[8:-2])
			
			# action = np.zeros(6, dtype=float)
			obs, reward, done, _ = env.step(action)
			true_Q_value += dis_gamma * reward
			dis_gamma *= args.discount
			if 'RNN' in args.policy_name:
				obs_vec = utils.fifo_data(obs_vec, obs)
		# env.render()
		true_Q_val_vec.append(true_Q_value)
	return np.mean(np.asarray(true_Q_val_vec))


# put the peg's initial state at a random position farther from the hole
def random_start(env):
	# x = np.random.uniform(-0.015, 0.015, 1)
	# y = np.random.uniform(-0.015, 0.015, 1)
	x = np.clip(np.random.normal(0, 0.01, 1), -0.012, 0.012)
	y = np.clip(np.random.normal(0, 0.01, 1), -0.012, 0.012)
	z = np.clip(np.random.normal(-0.01, 0.002, 1), -0.012, -0.008)
	rx = np.clip(np.random.normal(0, 0.015, 1), -0.015, 0.015)
	ry = np.clip(np.random.normal(0, 0.015, 1), -0.015, 0.015)
	rz = np.clip(np.random.normal(0, 0.015, 1), -0.015, 0.015)
	action = np.hstack([x, y, z, rx, ry, rz])
	print(action)
	for i in range(200):
		obs, reward, done, _ = env.step(action)
		env.render()
	time.sleep(0)
	# print('**************************** random init pos ******************************')
	return obs