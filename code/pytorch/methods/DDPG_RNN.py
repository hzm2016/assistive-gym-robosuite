import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_layer=1,
                 l1_hidden_dim=400, l2_hidden_dim=300):
        super().__init__()
        self.gru = nn.GRU(state_dim, l1_hidden_dim, hidden_layer, batch_first=True)
        self.l2 = nn.Linear(l1_hidden_dim, l2_hidden_dim)
        self.l3 = nn.Linear(l2_hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, l1_hidden_dim=300, hidden_layer=1):
        super(Critic, self).__init__()

        # Q1 architecture
        self.gru1 = nn.GRU(state_dim, l1_hidden_dim, hidden_layer, batch_first=True)
        self.l1 = nn.Linear(action_dim, 100)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        self.gru1.flatten_parameters()
        xg1, _ = self.gru1(x)
        xg1 = xg1[:, -1, :]
        u1 = F.relu(self.l1(u))
        xu1 = torch.cat([xg1, u1], 1)

        x1 = F.relu(self.l2(xu1))
        x1 = self.l3(x1)

        return x1


class DDPG_RNN(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(-1, state.shape[0], state.shape[1])).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        not_done = torch.FloatTensor(1 - d).to(device)
        reward = torch.FloatTensor(r).to(device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

    def load(self, filename, directory):
        actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
        self.actor.load_state_dict(torch.load(actor_path))
        actor_optimizer_path = glob.glob('%s/%s_actor_optimizer.pth' % (directory, filename))[0]
        self.actor_optimizer.load_state_dict(torch.load(actor_optimizer_path))
        critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
        self.critic.load_state_dict(torch.load(critic_path))
        critic_optimizer_path = glob.glob('%s/%s_critic_optimizer.pth' % (directory, filename))[0]
        self.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))
        print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
