import math
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ..methods import DDPG, TD3, SAC
from envs.abb.models import utils


class Solver(object):
    def __init__(self, args, env, project_path):
        self.args = args
        self.env = env
        
        self.file_name = ''
        self.project_path = project_path
        self.result_path = project_path + "results/robosuite"
        
        self.evaluations = []
        
        # Set seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # print('action_dim :::', env._action_dim)
        # print("obs :::", env._setup_observables())
        state_dim = env.observation_space.shape[0]
        print('state_dim', state_dim)
        action_dim = env.action_space.shape[0]
        print('action_dim', action_dim)
        print(env.action_space.high)
        max_action = float(env.action_space.high[0])
        
        # Initialize policy
        if 'DDPG' == args.policy_name:
            policy = DDPG.DDPG(args, state_dim, action_dim, max_action)
        elif 'SAC' == args.policy_name:
            policy = SAC.SAC(args, state_dim, action_dim, max_action, self.env.action_space)
        elif 'TD3' == args.policy_name:
            policy = TD3.TD3(args, state_dim, action_dim, max_action)
        else:
            policy = TD3.TD3(args, state_dim, action_dim, max_action)
        
        self.log_dir = '{}/{}/{}_{}_seed_{}'.format(self.result_path,
                                                    self.args.log_path,
                                                    self.args.policy_name,
                                                    self.args.env_name,
                                                    self.args.seed)
        print("---------------------------------------")
        print("Settings: %s" % self.log_dir)
        print("---------------------------------------")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # self.log_transfer_dir = '{}/{}_transfer/{}_{}_seed_{}'.format(self.result_path,
        #                                                               self.args.log_path,
        #                                                               self.args.policy_name,
        #                                                               self.args.env_name,
        #                                                               self.args.seed)
        # print("---------------------------------------")
        # print("Settings: %s" % self.log_transfer_dir)
        # print("---------------------------------------")
        # if not os.path.exists(self.log_transfer_dir):
        #     os.makedirs(self.log_transfer_dir)
        
        self.policy = policy
        self.replay_buffer = utils.ReplayBuffer()
        
        self.total_timesteps = 0
        self.pre_num_steps = self.total_timesteps
        
        self.best_reward = 0.0

        self.writer_train = SummaryWriter(logdir=self.log_dir)
        # self.writer_test = SummaryWriter(logdir=self.log_dir)

    def reset(self):
        self.obs = self.env.reset()
        self.episode_reward = 0
        self.episode_timesteps = 0
    
    def train_once(self):
        if self.total_timesteps != 0:
            self.writer_train.add_scalar('train_ave_reward', self.episode_reward, self.total_timesteps)

            self.policy.train(self.replay_buffer,
                              self.args.batch_size,
                              self.args.discount,
                              self.args.tau,
                              self.args.policy_noise,
                              self.args.noise_clip,
                              self.args.policy_freq)
    
    def eval_once(self):
        self.pbar.update(self.total_timesteps - self.pre_num_steps)
        self.pre_num_steps = self.total_timesteps
        
        # Evaluate episode
        if self.total_timesteps%self.args.eval_freq==0:
            # evaluate the policy for once
            avg_reward, avg_episode_steps = evaluate_policy(self.env, self.policy, self.args)
            
            self.evaluations.append(avg_reward)
            self.writer_train.add_scalar('test_ave_reward', avg_reward, self.total_timesteps)
            
            if self.best_reward < avg_reward:
                self.best_reward = avg_reward
                print("Best reward! Total T: %d Episode T: %d Reward: %f" %
                      (self.total_timesteps, self.episode_timesteps, avg_reward))
                self.policy.save(self.file_name, directory=self.log_dir)
                
            np.save(self.log_dir + "/test_accuracy", self.evaluations)
            utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
    
    def train(self):
        avg_reward,  _ = evaluate_policy(self.env, self.policy, self.args)
        self.evaluations = [avg_reward]
        
        self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
        
        if self.args.load_policy:
            self.policy.load(self.file_name + str(self.args.load_policy_idx), self.log_dir)
        
        done = False
        self.reset()
        while self.total_timesteps < self.args.max_timesteps:
            self.train_once()
            if done or self.episode_timesteps + 1 > self.args.max_episode_steps:
                print('done', done)
                print('total_timesteps', self.total_timesteps)
                print('episode_reward', self.episode_reward)
                self.eval_once()
                self.reset()
                done = False
            
            # Select action randomly or according to policy
            if self.total_timesteps < self.args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                if 'SAC' in self.args.policy_name:
                    action = self.policy.select_action(np.array(self.obs), eval=False)
                else:
                    action = self.policy.select_action(np.array(self.obs))
                
                if self.args.expl_noise != 0:
                    action = (action + np.random.normal(0, self.args.expl_noise,
                                                        size=self.env.action_space.shape[0])).clip(
                        self.env.action_space.low[0], self.env.action_space.high[0])
            
            new_obs, reward, done, _ = self.env.step(action)
            
            if self.args.render:
                self.env.render()

            self.episode_reward += reward
            
            done_bool = 0 if self.episode_timesteps + 1 == self.args.max_episode_steps else float(done)
            p = 1.0
            self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool, p))

            self.obs = new_obs
            self.episode_timesteps += 1
            self.total_timesteps += 1
        
        avg_reward, avg_episode_steps = evaluate_policy(self.env, self.policy, self.args)
        self.evaluations.append(avg_reward)
        
        if self.best_reward < avg_reward:
            self.best_reward = avg_reward
            print("Best reward! Total T: %d Episode T: %d Reward: %f" %
                  (self.total_timesteps, self.episode_timesteps, avg_reward))
            self.policy.save(self.file_name, directory=self.log_dir)
        
        if self.args.save_all_policy:
            self.policy.save(self.file_name + str(int(self.args.max_timesteps)), directory=self.log_dir)
        
        # if self.args.load_policy:
        #     np.save(self.log_transfer_dir + "/test_accuracy", self.evaluations)
        #     utils.write_table(self.log_transfer_dir + "/test_accuracy", np.asarray(self.evaluations))
        # else:
        
        np.save(self.log_dir + "/test_accuracy", self.evaluations)
        utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
        
        # # save the replay buffer
        if self.args.save_data:
            self.replay_buffer.save_buffer(self.log_dir + "/buffer_data")
        
        self.env.reset()
    
    def eval_only(self):
        self.evaluations = [evaluate_policy(self.env, self.policy, self.args)]
        
        self.writer_test = SummaryWriter(logdir=self.log_dir + '_test')
        self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
        
        if self.args.load_policy:
            self.policy.load(self.file_name + str(self.args.load_policy_idx), self.log_dir)
        
        done = False
        safe_or_not = True
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
            if 'SAC' in self.args.policy_name:
                action = self.policy.select_action(np.array(self.obs), eval=False)
            else:
                action = self.policy.select_action(np.array(self.obs))
            
            new_obs, reward, done, _ = self.env.step(action)

            self.episode_reward += reward
            
            done_bool = 0 if self.episode_timesteps + 1 == self.args.max_episode_steps else float(done)
            
            self.obs = new_obs
            self.episode_timesteps += 1
            self.total_timesteps += 1
        
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


def evaluate_policy(env, policy, args, eval_episodes=5):
    avg_reward = 0.
    avg_episode_steps = 0
    for _ in range(eval_episodes):
        print('eval_episodes', eval_episodes)
        obs = env.reset()
        # obs, state, done = env.reset()
        done = False

        eval_episodes_steps = 0
        episode_states = []
        while not done and eval_episodes_steps < args.max_episode_steps:
            action = policy.select_action(np.array(obs))
    
            # obs, _, reward, done, safe_or_not = env.step(action)
            obs, reward, done, _ = env.step(action)
            episode_states.append(obs)
            avg_reward += reward
            avg_episode_steps += 1
            eval_episodes_steps += 1
        
    avg_reward /= eval_episodes
    avg_episode_steps /= eval_episodes
    print('eval_avg_reward', avg_reward)
    return avg_reward, avg_episode_steps
