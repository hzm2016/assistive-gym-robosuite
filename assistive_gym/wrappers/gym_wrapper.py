"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from gym.core import Env
from .wrapper import Wrapper
from tensorboardX import SummaryWriter as FileWriter
import time


class GymWrapper(Wrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, logdir, keys=None):
        
        # Run super method
        super().__init__(env=env)
        
        # Create name for gym
        # robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = type(self.env).__name__
        
        self.logdir = logdir
        
        # Get reward range
        # self.reward_range = (0, self.env.reward_scale)
        self.metadata = None
        self.reward_range = (-float(1000), float(1000))
        self.keys = keys

        # Gym specific attributes
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        print("obs", obs)
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
    
        self.action_space = self.env.action_space
        print("action_space :", self.action_space)
 
        self.episode_info = dict()
        self.total_steps = 0
        self.episode_reward = 0.0
        
        self.writer = FileWriter(logdir)
        self._max_episode_steps = 500
        
        # print('obs_dim :', self.obs_dim, self.action_space.shape[0])
        # self.replay_buffer = ReplayBuffer(int(self.obs_dim), int(self.action_space.shape[0]), max_size=max_size)
        self.state = None
        self.next_state = None

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        self.state = self._flatten_obs(ob_dict)
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        
        self.episode_reward += reward
        # for key in self.keys:
        #     if key not in info:
        #         break
        #     if key in self.episode_info:
        #         self.episode_info[key].append(info[key])
        #     else:
        #         self.episode_info[key] = [info[key]]

        if done:
            self.writer.add_scalar('train_episode_reward', self.episode_reward, self.total_steps)
            self.episode_reward = 0
            
            # Clear the episode_info dictionary
            self.episode_info = dict()
        
        self.next_state = self._flatten_obs(ob_dict)
        self.total_steps += 1
        # self.replay_buffer.add(self.state, action, self.next_state, reward, done)
        self.state = self.next_state.copy()
        return self._flatten_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def _flatten_obs(self, obs_dict):
        """
            Filters keys of interest out and concatenate the information.
        """
        # ob_lst = []
        # ob_lst.append(obs_dict["robot"])
        # ob_lst.append(np.cos(obs_dict["robot_joint_angles"]))
        # ob_lst.append(np.sin(obs_dict["robot_joint_angles"]))
        # return np.concatenate(ob_lst)
        return obs_dict

    def _flatten_reward(self, reward, done, info):
    
        # return reward['robot'], done['__all__'], info['robot']
        return reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def close(self):
        """
        Closes the FileWriter and the underlying environment.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            
    def save_buffer(self):
        # self.replay_buffer.save(self.logdir + "/saved_buffer")
        pass
