"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from gym.core import Env
from envs.robosuite.robosuite.wrappers import Wrapper
from tensorboardX import SummaryWriter as FileWriter
from code.pytorch.PLAS.utils import ReplayBuffer
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

    def __init__(self, env, logdir, max_size, keys=None):
        
        # Run super method
        super().__init__(env=env)
        
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__
        
        self.logdir = logdir
        
        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        self.episode_info = dict()
        self.total_steps = 0
        self.episode_reward = 0.0
        self.writer = FileWriter(logdir)

        self._max_episode_steps = 500
        
        print('obs_dim :', self.obs_dim, self.action_space.shape[0])
        self.replay_buffer = ReplayBuffer(int(self.obs_dim), int(self.action_space.shape[0]), max_size=max_size)
        self.state = None
        self.next_state = None

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        # print("ob_dict :", ob_dict)
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
        # time.sleep(.002)
        
        self.episode_reward += reward
        for key in self.keys:
            if key not in info:
                break
            if key in self.episode_info:
                self.episode_info[key].append(info[key])
            else:
                self.episode_info[key] = [info[key]]

        if done:
            self.writer.add_scalar('train_episode_reward', self.episode_reward, self.total_steps)
            self.episode_reward = 0
            
            # Clear the episode_info dictionary
            self.episode_info = dict()
        
        self.next_state = self._flatten_obs(ob_dict)
        self.total_steps += 1
        self.replay_buffer.add(self.state, action, self.next_state, reward, done)
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
        self.replay_buffer.save(self.logdir + "/saved_buffer")