import numpy as np
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
import torch.nn.functional as F
if torch.cuda.is_available():
	torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.l4 = nn.Linear(state_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, action_dim)

		self.l7 = nn.Linear(state_dim, 400)
		self.l8 = nn.Linear(400, 300)
		self.l9 = nn.Linear(300, action_dim)

		self.max_action = max_action

	def forward(self, x):
		x1 = F.relu(self.l1(x))
		x1 = F.relu(self.l2(x1))
		x1 = self.max_action * torch.tanh(self.l3(x1))

		x2 = F.relu(self.l4(x))
		x2 = F.relu(self.l5(x2))
		x2 = self.max_action * torch.tanh(self.l6(x2))

		x3 = F.relu(self.l7(x))
		x3 = F.relu(self.l8(x3))
		x3 = self.max_action * torch.tanh(self.l9(x3))
		return torch.stack([x1, x2, x3], dim=2)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(xu))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2


	def Q1(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1 


class AAC(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.max_action = max_action
		self.it = 0

	def select_optimal_action(self, state, actor, critic):
		action_tensor = actor(state)
		Q_val_array = torch.zeros(state.shape[0], action_tensor.shape[-1])
		for i in range(action_tensor.shape[-1]):
			Q1, Q2 = critic(state, action_tensor[:, :, i])
			Q_val_array[:, i] = 0.5 * (Q1.squeeze() + Q2.squeeze())
		policy_idx = torch.argmax(Q_val_array, dim=-1, keepdim=False)
		return action_tensor[torch.arange(0, action_tensor.shape[0]), :, policy_idx]

	def calc_actor_discrepancy(self, state, actor):
		action_tensor = actor(state)
		action_discrepancy = action_tensor - torch.mean(action_tensor, dim=-1, keepdim=True)
		action_mse = torch.mean(torch.sum(action_discrepancy**2, dim=-1))
		return action_mse/(torch.abs(action_mse) + 1)

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		action = self.select_optimal_action(state, self.actor, self.critic)
		return action.cpu().data.numpy().flatten()

	def cal_estimate_value(self, replay_buffer, eval_states=10000):
		x, _, u, _, _ = replay_buffer.sample(eval_states)
		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		Q1, Q2 = self.critic(state, action)
		# target_Q = torch.mean(torch.min(Q1, Q2))
		Q_val = 0.5 * (torch.mean(Q1) + torch.mean(Q2))
		return Q_val.detach().cpu().numpy()

	def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005,
			  policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		self.it += 1
		# Sample replay buffer
		x, y, u, r, d = replay_buffer.sample(batch_size)
		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		next_state = torch.FloatTensor(y).to(device)
		done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)

		# Select action according to policy and add clipped noise
		noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
		noise = noise.clamp(-noise_clip, noise_clip)
		next_action = self.select_optimal_action(
			next_state, self.actor_target, self.critic_target)
		next_action = (next_action + noise).clamp(
			-self.max_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.critic_target(next_state, next_action)

		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + (done * discount * target_Q).detach()

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) \
					  - 0.1 * F.mse_loss(current_Q1, current_Q2)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.it % policy_freq == 0:
			# Compute actor loss
			action = self.select_optimal_action(next_state, self.actor, self.critic)
			current_Q1, current_Q2 = self.critic(state, action)
			actor_loss = - 0.5 * (current_Q1 + current_Q2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
		self.actor.load_state_dict(torch.load(actor_path))
		critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
		print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
		self.critic.load_state_dict(torch.load(critic_path))
