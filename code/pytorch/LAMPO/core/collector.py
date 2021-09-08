import numpy as np
from code.pytorch.LAMPO.core.task_interface import TaskInterface
from code.pytorch.LAMPO.core.model import RLModel


class RunModel:

    def __init__(self, task: TaskInterface, rl_model: RLModel, dense_reward=False):
        self.task = task
        self._rl_model = rl_model
        self.dense_reward = dense_reward

    def collect_samples(self, n_episodes=50, noise=True, isomorphic_noise=False):
        success_list = []
        reward_list = []
        latent = []
        cluster = []
        observations = []
        parameters = []

        for i in range(n_episodes):
            # env reset
            self.task.reset()
            
            # get context
            context, _, _, _ = self.task.read_context()
            observations.append(context)
            w, z, k = self._rl_model.generate_full(np.expand_dims(context, 0),
                                                   noise=noise,
                                                   isomorphic_noise=isomorphic_noise)
            
            print("parameters :::", w)
            parameters.append(w)
            latent.append(z)
            cluster.append(k)
            
            success, tot_reward = self.task.send_movement(w[1:], w[0])
            print(success, tot_reward)
            success_list.append(success)

            if self.dense_reward:
                reward_list.append(tot_reward)
            else:
                reward_list.append(success)

        print("-"*50)
        print("Total reward", np.mean(reward_list))
        print("-"*50)
        return np.array(success_list), np.array(reward_list), np.array(parameters), np.array(latent),\
               np.array(cluster), np.array(observations)


class RunModelMujoco:
    
    def __init__(self, task: TaskInterface, rl_model: RLModel, dense_reward=False):
        self.task = task
        self._rl_model = rl_model
        self.dense_reward = dense_reward
    
    def collect_samples(self, n_episodes=50, noise=True, isomorphic_noise=False):
        success_list = []
        reward_list = []
        latent = []
        cluster = []
        observations = []
        parameters = []
        
        for i in range(n_episodes):
            
            # env reset
            self.task.reset()
            
            # get context
            context = self.task.read_context()
            print("context :", context, "process :", np.expand_dims(context, 0))
            
            # observation : context
            observations.append(context)
            w, z, k = self._rl_model.generate_full(np.expand_dims(context, 0),
                                                   noise=noise,
                                                   isomorphic_noise=isomorphic_noise)

            # print("parameters :", w)
            parameters.append(w)
            latent.append(z)
            cluster.append(k)
            
            success, _, tot_reward = self.task.send_movement(w)
            # print("success :", success, "reward :", tot_reward)
            
            success_list.append(success)
            
            if self.dense_reward:
                reward_list.append(tot_reward)
            else:
                reward_list.append(success)
        
        print("-" * 50)
        print("Total reward", np.mean(reward_list))
        print("-" * 50)
        return np.array(success_list), np.array(reward_list), np.array(parameters), np.array(latent), \
               np.array(cluster), np.array(observations)


class RunModelPybullet:
    
    def __init__(self, task: TaskInterface, rl_model: RLModel, dense_reward=False):
        self.task = task
        self._rl_model = rl_model
        self.dense_reward = dense_reward
    
    def collect_samples(self, n_episodes=50, noise=True, isomorphic_noise=False):
        success_list = []
        reward_list = []
        
        observations = []
        parameters = []
        
        latent = []
        cluster = []
        
        sort = False
        for i in range(n_episodes):
            print("Episode :", i)
            
            # env reset
            self.task.reset()
            
            # get context
            context = self.task.read_context()
            print("context :", context, "process :", np.expand_dims(context, 0))
            
            # observation : context
            observations.append(context)
            w, z, k = self._rl_model.generate_full(np.expand_dims(context, 0),
                                                   noise=noise,
                                                   isomorphic_noise=isomorphic_noise)
            
            print("parameters :", w)
            parameters.append(w)
            latent.append(z)
            cluster.append(k)
            
            reward, _, info = self.task.send_movement(w)
            # print("reward :", reward, "success :", info["task_success"])
            
            success_list.append(info["task_success"])
            reward_list.append(reward)
        
        if sort:
            index_sort = np.argsort(np.array(reward_list))[::-1]
        
        # print("-" * 50)
        # print("Total reward", np.mean(reward_list), "Success Rate :", np.sum(success_list)/n_episodes)
        # print("-" * 50)
        return np.array(reward_list), np.array(success_list), np.array(parameters), np.array(latent), \
               np.array(cluster), np.array(observations)
