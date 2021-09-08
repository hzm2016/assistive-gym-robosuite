import numpy as np
import os
import torch
import math
from envs.abb.models import utils
from tqdm import tqdm

import time
import copy as cp

from ..methods import DDPG, TD3, SAC, AAC


class Solver(object):
    def __init__(self, args, env, project_path):
        self.args = args
        self.env = env
        self.file_name = ''
        self.project_path = project_path
        self.result_path = project_path + "results/runs/real"

        self.evaluations = []
        self.estimate_Q_vals = []
        self.true_Q_vals = []

        # Set seeds
        # self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        print('state_space_dim :::', state_dim)
        print('action_space_dim :::', action_dim)

        # Initialize policy
        if 'TD3' == args.policy_name:
            policy = TD3.TD3(state_dim, action_dim, max_action)
        elif 'DDPG' == args.policy_name:
            policy = DDPG.DDPG(state_dim, action_dim, max_action)
        elif 'SAC' == args.policy_name:
            policy = SAC.SAC(args, state_dim, action_dim, max_action, self.env.action_space)
        elif 'AAC' == args.policy_name:
            policy = AAC.AAC(state_dim, action_dim, max_action)
        else:
            exit("Please give right control algorithm !!!")

        self.policy = policy
        self.replay_buffer = utils.ReplayBuffer()

        self.total_timesteps = 0
        self.best_reward = 0.0  # try to find the best policy :::: with best reward

        self.log_dir = '{}/{}/{}_{}_seed_{}'.format(self.result_path,
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

        # Loaded trained policy
        if self.args.load_policy:
            self.policy.load(self.file_name, self.args.load_path)

    def reset(self):
        """
            reset real-world env
        """
        self.obs, _, _ = self.env.reset()
        self.episode_reward = 0
        self.episode_timesteps = 0
        done = False
        safe_or_not = True
        return done, safe_or_not

    def train(self):
        """ evaluation performance """
        self.evaluations_reward = []
        self.evaluations_time = []
        self.evaluations_actions = []  # impedance parameters
        self.evaluations_im_actions = []  # execute action by robot
        self.evaluations_states = []

        """ training performance """
        self.training_reward = []
        self.training_time = []
        self.training_states = []
        self.training_im_actions = []

        self.pbar = tqdm(total=self.args.max_training_episodes, initial=0, position=0, leave=True)

        EPS_START = 0.25
        EPS_END = 0.05
        EPS_DECAY = self.args.max_timesteps

        for i in range(self.args.max_training_episodes):
            self.pbar.update(i)
            
            if i % self.args.eval_episodes == 0:
                self.eval_once()
        
            done, safe_or_not = self.reset()
            self.args.expl_noise *= 0.95
          
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * self.total_timesteps / EPS_DECAY)

            epi_states = []
            epi_actions = []
            while self.episode_timesteps + 1 < self.args.max_episode_steps:
                if done:
                    print('Done successfully:::::::::::::::::::::::::', done)
                    break

                if safe_or_not is False:
                    print('Safe_is_not ::::::::::::::::::::::::::::::', safe_or_not)
                    break

                """ Select action randomly or according to policy """
                if self.total_timesteps < self.args.start_timesteps:
                    action = self.env.action_space.sample()
                    p = 1
                    self.option = np.random.randint(self.args.option_num)
                    self.next_option = np.random.randint(self.args.option_num)
                else:
                    if 'SAC' in self.args.policy_name:
                        action = self.policy.select_action(np.array(self.obs), eval=False)
                    else:
                        action = self.policy.select_action(np.array(self.obs))

                    if self.args.expl_noise != 0:
                        action = (action +
                                  np.random.normal(0, self.args.expl_noise, size=self.env.action_space.shape[0])).clip(
                            self.env.action_space.low[0], self.env.action_space.high[0])

                # make sure with the changed environment
                new_obs, original_state, reward, done, safe_or_not, executeAction = self.env.step(action)
                # new_obs, original_state, reward, done, safe_or_not = self.env.step(action)
                
                done_bool = 0 if self.episode_timesteps + 1 == self.args.max_episode_steps else float(done)
                self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool, 1.))

                epi_states.append(cp.deepcopy(self.obs))
                epi_actions.append(cp.deepcopy(executeAction))
                self.obs = new_obs
                self.episode_reward += reward
                self.episode_timesteps += 1
                self.total_timesteps += 1

            print('total_timesteps ::::::::::::::::::::::::::', self.total_timesteps)
            print('episode_reward :::::::::::::::::::::::::::', self.episode_reward)
            self.training_reward.append(cp.deepcopy(self.episode_reward))
            self.training_time.append(cp.deepcopy(self.episode_timesteps))
            self.training_states.append(cp.deepcopy(epi_states))
            self.training_im_actions.append(cp.deepcopy(epi_actions))

            np.save(self.log_dir + "/train_reward", self.training_reward)
            np.save(self.log_dir + "/train_time", self.training_time)
            np.save(self.log_dir + "/train_states", self.training_states)
            np.save(self.log_dir + "/train_im_actions", self.training_im_actions)

            # convenient to check the results
            utils.write_table(self.log_dir + "/train_reward", np.asarray(self.training_reward))
            utils.write_table(self.log_dir + "/train_time", np.asarray(self.training_time))
            
            self.train_once()

        """ final evaluation """
        self.eval_once()
        self.reset()

        """ store buffer data ::: important for our offline RL """
        # if 'HRLACOP' == self.args.policy_name:
        #     self.replay_buffer_low.save_buffer(self.log_dir + "/option_buffer_data")
        # else:
        #     self.replay_buffer.save_buffer(self.log_dir + "/training_data")
        
        # avg_reward, avg_time, eval_actions, eval_states = evaluate_assembly_policy(self.env, self.policy, self.args)
        # self.evaluations_reward.append(avg_reward)
        # self.evaluations_time.append(avg_time)
        # self.evaluations_actions.append(eval_actions)
        # self.evaluations_states.append(eval_states)
        # print('evaluations_reward :::::::::::::::', self.evaluations_reward)
        # print('evaluations_time :::::::::::::::::', self.evaluations_time)

        # """ save test data """
        # np.save(self.log_dir + "/test_reward", self.evaluations_reward)
        # np.save(self.log_dir + "/test_time", self.evaluations_time)
        # np.save(self.log_dir + "/test_actions", self.evaluations_actions)
        # np.save(self.log_dir + "/test_states", self.evaluations_states)
        # utils.write_table(self.log_dir + "/test_reward", np.asarray(self.evaluations_reward))
        # utils.write_table(self.log_dir + "/test_time", np.asarray(self.evaluations_time))

        # if self.best_reward < avg_reward:
        #     self.best_reward = avg_reward
        #     print("Best reward! Total T: %d Episode T: %d Reward: %f" %
        #           (self.total_timesteps, self.episode_timesteps, avg_reward))

        """ save training data """
        # np.save(self.log_dir + "/train_reward", self.training_reward)
        # np.save(self.log_dir + "/train_time", self.training_time)
        # utils.write_table(self.log_dir + "/train_reward", np.asarray(self.training_reward))
        # utils.write_table(self.log_dir + "/train_time", np.asarray(self.training_time))

    def eval_only(self):
        # model_path_vec = glob.glob(self.args.load_path)
        # for model_path in model_path_vec:
        # self.policy.load("%s" % (self.file_name),
        #                  directory=self.args.load_path)
        self.policy.load(self.file_name, directory=self.args.load_path)
        eval_states = []
        eval_reward = []
        eval_time = []
        eval_action = []
        eval_im_action = []
        for _ in range(self.args.num_test_episodes):
            """ Reset environment """
            self.obs, original_state, _ = self.env.reset()
            self.option = np.random.randint(self.args.option_num)
            done = False
            time_step = 0
            safe_or_not = True
            start_time = time.time()
            avg_reward = 0.
            episode_states = []
            episode_actions = []
            episode_im_actions = []
            while not done and safe_or_not and time_step < self.args.max_episode_steps:
                """ obtain action from policy """
                if 'HRLACOP' == self.args.policy_name:
                    if (time_step % self.args.option_change == 0):
                        action, self.option = self.policy.select_action(np.array(self.obs),
                                                                        self.option,
                                                                        change_option=True)
                    else:
                        action, self.option = self.policy.select_action(np.array(self.obs),
                                                                        self.option,
                                                                        change_option=False)
                else:
                    action = self.policy.select_action(np.array(obs))

                obs, original_state, reward, done, safe_or_not, executeAction = self.env.step(action)
                episode_states.append(cp.deepcopy(original_state))

                time_step += 1
                avg_reward += reward
                episode_actions.append(cp.deepcopy(action))
                episode_im_actions.append(cp.deepcopy(action))

            eval_reward.append(cp.deepcopy(avg_reward))
            eval_time.append(cp.deepcopy(time.time() - start_time))
            eval_states.append(cp.deepcopy(episode_states))
            eval_action.append(cp.deepcopy(episode_actions))
            eval_im_action.append(cp.deepcopy(episode_im_actions))

        utils.write_table(self.log_dir + "/evaluation_reward", np.asarray(eval_reward))
        utils.write_table(self.log_dir + "/evaluation_time", np.asarray(eval_time))
        np.save(self.log_dir + "/evaluation_states", np.asarray(eval_states))
        np.save(self.log_dir + "/evaluation_actions", np.asarray(eval_action))
        np.save(self.log_dir + "/evaluation_im_actions", np.asarray(eval_im_action))

    def train_once(self):
        print(':::::::::::::::::::::: train once :::::::::::::::::::::')
        if self.total_timesteps != 0:
            # self.writer_train.add_scalar('ave_reward', self.episode_reward, self.total_timesteps)
            for j in range(self.args.num_training_steps):
                self.policy.train(self.replay_buffer,
                                  self.args.batch_size,
                                  self.args.discount,
                                  self.args.tau,
                                  self.args.policy_noise,
                                  self.args.noise_clip,
                                  self.args.policy_freq
                                  )

    def eval_once(self):
        print(':::::::::::::::::::::: evaluation once :::::::::::::::::::::')
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
        print('evaluations_reward :::::::::::::::', self.evaluations_reward)
        print('evaluations_time :::::::::::::::::', self.evaluations_time)

        """ save test data numpy """
        np.save(self.log_dir + "/test_reward", self.evaluations_reward)
        np.save(self.log_dir + "/test_time", self.evaluations_time)
        np.save(self.log_dir + "/test_actions", self.evaluations_actions)
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


def evaluate_assembly_policy(env, policy, args):
    """
        Runs policy for X episodes and returns average reward
    """
    avg_reward = 0.
    eval_actions = []
    eval_im_actions = []
    eval_states = []
    start_time = time.time()
    for _ in range(args.num_eval_episodes):
        obs, state, done = env.reset()
        done = False
        episode_step = 0
        epi_actions = []
        epi_im_actions = []
        epi_states = []
        while not done and episode_step < args.max_episode_steps:
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
    avg_time = (time.time() - start_time) / args.num_eval_episodes
    avg_reward /= args.eval_episodes
    return avg_reward, avg_time, eval_actions, eval_states, eval_im_actions, _