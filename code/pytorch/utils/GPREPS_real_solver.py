import numpy as np
from envs.abb.models import utils
from code.pytorch.methods import TD3
from ECPL_pytorch.argmin import argmin_g as argmin
from envs.abb.env_abb_assembly import env_assembly_search
from algorithms.calculations import calculation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from code.pytorch.Regression.GMRbasedGP.utils.gmr import Gmr


class Old_Solver(object):
    def __init__(self,
                 env):
        
        # #####################  Hyper Parameters  ######################
        self.K = 15  # 上层policy训练循环总数
        self.N = 10  # 在上层policy的一个训练周期中，下层RL训练，改变context参数的次数
        self.n = 1  # 下层RL训练，每改变一次context参数，执行RL的episode数
        self.d = 1  # 下层RL每d个step更新一次网络参数
        self.M = 5  # 在上层policy的一个训练周期中，通过model获得artificial trajectory的过程中，改变context参数的次数
        self.L = 20  # 每改变一次context参数，通过model获得artificial trajectory的次数
        
        # Initialize assembly environment
        self.env = env
        self.init_state = np.hstack(
            [np.array([0, 0, 0, 0, 0, 0]), np.hstack([self.env.set_initial_pos, self.env.set_initial_euler])])
        self.state_bound = np.array([10, 10, 10, 1, 1, 1, 0.2, 0.2, 7, 0.10, 0.10, 0.10])
        # self.action_bound = np.array([0.4, 0.15, 0.26, 0.09, 0.09, 0.10])
        
        # # Set seeds
        # torch.manual_seed(args.seed)
        # np.random.seed(args.seed)
        
        # RL parameters
        self.MAX_EP_STEPS = 200  # RL的最大步数
        self.batch_size = 64
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.RL_buffer_size = 6000
        self.state_dim = self.env.observation_dim
        self.action_dim = self.env.action_dim
        self.max_RL_action = 1.
        self.min_RL_action = -1.
        
        # Initialize RL policy
        self.done = False
        self.safe = True
        self.pd = True
        self.var = 0.3
        
        'open TD_3'
        "Lower-level RL policy"
        # path_TD_3 = 'C:/Users/liucc/Desktop/RL_assembly/TD3_100/TD_3.txt'
        # file_TD_3 = open(path_TD_3, 'rb')
        # self.policy = pickle.load(file_TD_3)
        self.policy = TD3.TD3(self.state_dim,
                              self.action_dim,
                              self.max_RL_action)
        
        " Replay buffer for lower-level policy "
        # path_replay_buffer = 'C:/Users/liucc/Desktop/RL_assembly/TD3_100/replay_buffer.txt'
        # file_replay_buffer = open(path_replay_buffer, 'rb')
        # self.replay_buffer = pickle.load(file_replay_buffer)
        self.replay_buffer = utils.ReplayBuffer(self.RL_buffer_size)
        
        self.episode_timesteps = 0
        self.total_timesteps = 0
        self.episode_reward = 0
        self.episode_number = 0
        
        " Contextual parameters "
        self.context_dim = 3  # 速度和深度系数共3维
        self.contextual_action_dim = 2  # Kp and Kd
        self.contextual_action_bound = 0.01  # Kp和Kd的范围暂定为0.1
        buffer_size = 60
        
        # Initialize Contextual Policy
        self.replay_buffer_model = utils.ReplayBuffer(buffer_size)
        self.gpreps = GPREPS(self.context_dim, self.contextual_action_dim, self.M, self.contextual_action_bound)
        # self.r_model = R_MODEL(self.policy, self.env, self.context_dim, self.contextual_action_dim, self.observation_dim, self.action_dim, self.MAX_EP_STEPS)
        # self.s_model = S_MODEL()
        
        'temp data'
        self.Reward = []
        self.State = []
        self.IM_Action = []
        self.R = []
    
    def train_once(self):
        if self.total_timesteps >= 200:
            self.policy.train(self.replay_buffer,
                              self.batch_size,
                              self.discount,
                              self.tau,
                              self.policy_noise)
            print('policy trained')
        else:
            pass
    
    def RL_reset(self):
        self.env.reset()
        self.state = self.env.start_noise()
        self.state = self.state - self.init_state
        self.done = False
        self.safe = True
        self.episode_reward = 0
        self.episode_timesteps = 0
    
    # u(z) 暂定s[速度系数, 深度系数]为0.6~1.0之间随机量
    def __get_z(self):
        depth = np.random.rand(1)[0] * 2 - 1  # -1~1之间随机数
        z = (np.random.rand(3) - 0.5) / 100 + np.array([0, depth, 0])
        self.z_ = np.array([0, 0, 0])
        return z
    
    # execute reinforcement learning and store data for model training
    def train(self):
        
        # n training cycles
        self.gpreps.memory = []
        R = 0
        
        for i in range(self.N):
            
            z = self.__get_z()
            print('context parameter z:', z)
            self.env.set_context(z)
            w = self.gpreps.choose_action(z)[0]
            print('contextual action w:', w)
            self.env.set_pd(w)
            
            self.episode_number = 0
            self.var = 0.3
            
            # Start DDPG training
            while self.episode_number < self.n:
                self.RL_reset()
                
                while self.episode_timesteps < self.MAX_EP_STEPS:
                    # 'train RL policy'
                    # if self.episode_timesteps % self.d == 0:
                    #     self.train_once()
                    
                    print('Pose:', self.state[6:9], 'Force:', self.state[:3])
                    action = self.policy.select_action(np.array(self.state / self.state_bound))
                    # print('RL action:', action)
                    # action = np.random.normal(action, self.var).clip(self.min_RL_action, self.max_RL_action)
                    # action = np.array([0, 0, 0, 0, 0, 0])
                    
                    'record state'
                    self.State.append(np.hstack([np.array([self.episode_timesteps]), self.state]))
                    
                    _, new_state, reward, self.done, self.safe, im_action = self.env.step(action)
                    new_state = new_state - self.init_state
                    print('Im_action:', im_action)
                    print('--------', 'Episode:', self.episode_number, 'STEP::', self.episode_timesteps, 'Reward:',
                          reward, '--------')
                    
                    'record im_action'
                    self.IM_Action.append(im_action)
                    
                    # store data into buffer
                    self.replay_buffer.add((self.state / self.state_bound, new_state / self.state_bound, action, reward,
                                            self.done, 0))  # for RL training
                    
                    self.episode_reward += reward
                    self.state = new_state.copy()
                    self.episode_timesteps += 1
                    self.total_timesteps += 1
                    if self.var >= 0.01:
                        self.var *= 0.999
                    
                    if self.done:
                        self.episode_number += 1
                        if self.episode_timesteps == self.MAX_EP_STEPS:
                            self.Reward.append([z, w, self.episode_timesteps, self.episode_reward, 'Unfinished'])
                        elif self.safe == False:
                            self.Reward.append([z, w, self.episode_timesteps, self.episode_reward, 'Failed'])
                        else:
                            self.Reward.append([z, w, self.episode_timesteps, self.episode_reward, 'Successful'])
                        print('Episode', i, 'Step', self.episode_timesteps, 'Total reward:', self.episode_reward,
                              'var:', self.var)
                        R += self.episode_reward
                        break
                self.gpreps.store_data(z, w / self.contextual_action_bound, self.episode_reward)
        
        self.R.append(R / self.N)
        np.save('C:/Users/Wenhao Yang/Desktop/TD3_control_reward.npy', self.R)
        np.save('C:/Users/Wenhao Yang/Desktop/TD3_control_reward.npy', self.Reward)
        np.save('C:/Users/Wenhao Yang/Desktop/TD3_control_state.npy', self.State)
        np.save('C:/Users/Wenhao Yang/Desktop//TD3_control_action.npy', self.IM_Action)
    
    # main function for upper level training
    def contextual_main(self):
        for k in range(self.K):
            print('Training cycle', k)
            # run RL and collect data
            print('Running RL...')
            self.train()
            
            # train reward and context model
            # print('Training GMR models...')
            # self.r_model.train_reward_model(self.replay_buffer_model)
            # self.s_model.train_context_model()
            
            # Predict Rewards and Store Data
            # print('Artificial trajectories')
            # for j in range(self.M):
            #     # Predict Rewards
            #     R = 0.
            #     Z = self.__get_z()
            #     W = self.gpreps.choose_action(Z)
            #
            #     # Predict L Trajectories
            #     for l in range(self.L):
            #         R += self.r_model.trajectory(Z, W)
            #     reward = R / self.L
            #     print('Artificial running cycle', j+1, 'ra:', Z[0], 'rd:', Z[1], 'reward:', reward)
            #
            #     # Construct Artiﬁcial Dataset D
            #     self.gpreps.store_data(Z, W, reward)
            
            # # Sample and Update Policy
            # S_ = self.z_average
            if k >= 2:
                self.gpreps.learn(self.z_)
            print('************ Training cycle finished **************')
            print('')
            
            
class Solver(object):
    def __init__(self):
        # #####################  Hyper Parameters  ######################
        self.K = 34  # 上层policy训练循环总数
        self.N = 5  # 在上层policy的一个训练周期中，下层RL训练，改变context参数的次数
        self.n = 1  # 下层RL训练，每改变一次context参数，执行RL的episode数
        self.d = 1  # 下层RL每d个step更新一次网络参数
        self.M = 1  # 在上层policy的一个训练周期中，通过model获得artificial trajectory的过程中，改变context参数的次数
        self.L = 10  # 每改变一次context参数，通过model获得artificial trajectory的次数

        # Initialize assembly environment
        self.env = env_assembly_search()
        self.init_state = np.hstack([np.array([0,0,0,0,0,0]), np.hstack([self.env.set_initial_pos, self.env.set_initial_euler])])
        self.action_bound = np.array([0.23, 0.23, 0.75, 0.06, 0.06, 0.03])
        self.dpos_bound = np.array([0.190, 0.157, 0.745, 0.110, 0.086, 0.012])
        self.force_bound = np.array([9.39, 9.32, 8.70, 1.00, 0.87, 0.92])
        self.pos_bound = np.array([1.96, 1.99, 23.93, 0.009, 0.06, 0.010])
        self.state_bound = np.hstack([self.force_bound, self.pos_bound])

        # # Set seeds
        # torch.manual_seed(args.seed)
        # np.random.seed(args.seed)

        # RL parameters
        self.MAX_EP_STEPS = 300  # RL的最大步数
        self.batch_size = 64
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        RL_buffer_size = 6000
        state_dim = self.env.observation_dim
        action_dim = self.env.action_dim
        self.max_RL_action = 1.
        self.min_RL_action = -1.

        # Initialize RL policy
        self.done = False
        self.safe = True
        self.pd = True
        self.var = 0.3

        'open TD_3'
        self.policy = TD3.TD3(state_dim, action_dim, self.max_RL_action)
        'open replay buffer'
        self.replay_buffer = utils.ReplayBuffer(RL_buffer_size)

        self.episode_timesteps = 0
        self.total_timesteps = 0
        self.episode_reward = 0
        self.episode_number = 0

        # Contextual parameters
        self.context_dim = 1  # 速度和深度系数共3维
        self.contextual_action_dim = 4  # Kp and Kd
        self.z_ = np.array([175])
        self.contextual_action_bound = np.array([0.008, 0.008, 0.004, 0.001])
        buffer_size = 10

        # Initialize Contextual Policy
        self.replay_buffer_model = utils.ReplayBuffer(buffer_size)
        self.gpreps = GPREPS(self.context_dim, self.contextual_action_dim, buffer_size, self.z_, self.contextual_action_bound)

        # data efficient hrl
        self.replay_buffer_high = utils.ReplayBufferHighLevel()
        self.replay_buffer_low = utils.ReplayBufferOption()

        'temp data'
        self.Reward = []
        self.State = []
        self.IM_Action = []
        self.R = []

    def train_once(self):
        if self.total_timesteps >= 300:
            self.policy.train(self.replay_buffer, self.batch_size, self.discount, self.tau, self.policy_noise)
            print('policy trained')
        else:
            pass

    def RL_reset(self):
        self.env.reset()
        self.env.start_noise()
        self.state = self.env.state
        self.state = self.state - self.init_state
        self.done = False
        self.safe = True
        self.episode_reward = 0
        self.episode_timesteps = 0

    # 暂定目标装配完成步数为z, 取值100-250随机值
    def __get_z(self):
        z = np.random.randint(0, 4, 1) * 50 + 100
        self.z_ = np.array([175])
        return z

    # execute reinforcement learning and store data for model training
    def train(self):

        "n training cycles"
        R = 0

        for i in range(self.N):

            z = self.__get_z()
            print('context parameter z:', z)
            self.env.set_context(z)
            w = self.gpreps.choose_action(z/self.z_)[0]
            print('contextual action w:', w)
            self.env.set_pd(w)

            self.episode_number = 0
            # self.var = 0.3

            "Start DDPG training"
            while self.episode_number < self.n:
                self.RL_reset()

                while self.episode_timesteps < self.MAX_EP_STEPS:
                    'train RL policy'
                    if self.episode_timesteps % self.d == 0:
                        self.train_once()

                    print('Pose:', self.state[6:9], 'Force:', self.state[:3])
                    action = self.policy.select_action(np.array(self.state/self.state_bound))
                    # print('RL action:', action)
                    # action = np.random.normal(action, self.var).clip(self.min_RL_action, self.max_RL_action)
                    # action = np.array([0, 0, 0, 0, 0, 0])

                    'record state'
                    self.State.append(np.hstack([np.array([self.episode_timesteps]), self.state]))

                    _, new_state, reward, self.done, self.safe, im_action = self.env.step(action)
                    new_state = new_state - self.init_state
                    print('Im_action:', im_action)
                    print('--------', 'Episode:', i*self.n+self.episode_number, 'STEP::', self.episode_timesteps, 'Reward:', reward, '--------')

                    'record im_action'
                    self.IM_Action.append(im_action)

                    'store data into buffer'
                    self.replay_buffer.add((self.state/self.state_bound, new_state/self.state_bound, action, reward, self.done, 0))  # for RL training
                    self.replay_buffer_model.add((self.state, new_state, im_action))  # for model training

                    'update state parameters'
                    self.episode_reward += reward
                    self.state = new_state.copy()
                    self.episode_timesteps += 1
                    self.total_timesteps += 1
                    if self.var >= 0.01:
                        self.var *= 0.999

                    if self.done:
                        if self.episode_timesteps == self.MAX_EP_STEPS - 1:
                            self.Reward.append([z, w, self.episode_timesteps,  self.episode_reward, 'Unfinished'])
                            print('Episode', i * self.n + self.episode_number, 'Step', self.episode_timesteps, 'Total reward:', self.episode_reward, 'Unfinished')
                        elif not self.safe:
                            self.Reward.append([z, w, self.episode_timesteps, self.episode_reward, 'Failed'])
                            print('Episode', i * self.n + self.episode_number, 'Step', self.episode_timesteps, 'Total reward:', self.episode_reward, 'Failed')
                        else:
                            self.Reward.append([z, w, self.episode_timesteps, self.episode_reward, 'Successful'])
                            print('Episode', i * self.n + self.episode_number, 'Step', self.episode_timesteps, 'Total reward:', self.episode_reward, 'Successful')

                        self.episode_number += 1
                        R += self.episode_reward
                        break
                self.gpreps.store_data(z/self.z_, w/self.contextual_action_bound, self.episode_reward)

        self.R.append(R / self.N)
        np.save('cycle_R.npy', self.R)
        np.save('Contextual_Policy_Search_reward.npy', self.Reward)
        np.save('Contextual_Policy_Search_state.npy', self.State)
        np.save('Contextual_Policy_Search_action.npy', self.IM_Action)

    # main function for upper level training
    def contextual_main(self):
        "Load initial data"
        Aa = np.load('model_based_17x10 (2)/Contextual_Policy_Search_Aa.npy', allow_pickle=True)
        reward = np.load('model_based_17x10 (2)/Contextual_Policy_Search_reward.npy', allow_pickle=True)
        state = np.load('model_based_17x10 (2)/Contextual_Policy_Search_state.npy', allow_pickle=True)
        action = np.load('model_based_17x10 (2)/Contextual_Policy_Search_action.npy', allow_pickle=True)
        for i in range(30):
            z = reward[i][0]
            w = reward[i][1]
            r = reward[i][3]
            self.gpreps.store_data(z/self.z_, w/self.contextual_action_bound, r)
            self.Reward.append(reward[i])
        self.gpreps.learn(self.z_/self.z_)
        # self.gpreps.a = Aa[-1][:, :1]
        # self.gpreps.A = Aa[-1][:, 1:]
        print('Episode imported, size:', len(self.Reward), len(self.gpreps.memory))

        t = 0
        for i in range(len(state)):
            if state[i][0] == 0:
                t += 1
            if t == 31:
                break
            self.State.append(state[i])
            self.IM_Action.append(action[i])
        print('State data imported, size:', len(self.State))

        "main"
        for k in range(self.K):
            print('Training cycle', k)
            # run RL and collect data
            print('Running RL...')
            self.train()

            # 'Sample and Update Policy'
            # if k >= 1:
            #     # train_model = True
            #     '第0,1,3,5,7,9...周期拟合forward model'
            #     if k % 4 == 1:
            #         train_model = True
            #     else:
            #         train_model = False
            #     self.gpreps.artificial_trajectory(self.State, self.IM_Action, self.L, self.policy, train_model)
            if k >= 0:
                self.gpreps.learn(self.z_/self.z_)
            print('************ Training cycle finished **************')
            print('')


class GPREPS(object):
    def __init__(self, z_dim, w_dim, memory_dim, z_, w_bound):
        # initialize parameters
        self.memory = []
        self.pointer = 0
        self.w_dim, self.z_dim, self.memory_dim, self.z_, self.w_bound = w_dim, z_dim, memory_dim, z_, w_bound

        # build actor
        self.a = np.zeros((w_dim, 1), dtype=np.float32)
        self.a[0][0] = -0.1
        self.a[1][0] = 0.1
        self.A = np.zeros((w_dim, z_dim), dtype=np.float32)
        self.COV = np.eye((w_dim), dtype=np.float32) / 10
        self.COV_index = 50
        self.Aa = []
        print('initial COV matrix:')
        print(self.COV)

        # impedance bound
        self.action_bound = np.array([0.23, 0.23, 0.75, 0.06, 0.06, 0.03])
        self.dpos_bound = np.array([0.190, 0.157, 0.745, 0.110, 0.086, 0.012])
        self.force_bound = np.array([9.39, 9.32, 8.70, 1.00, 0.87, 0.92])
        self.pos_bound = np.array([1.96, 1.99, 23.93, 0.009, 0.06, 0.010])
        self.state_bound = np.hstack([self.force_bound, self.pos_bound])

    def choose_action(self, z):
        z = np.array([z])
        u = self.a + np.dot(self.A, z.transpose())
        u = u.transpose()[0]
        return np.clip(np.random.multivariate_normal(mean=u, cov=self.COV, size=1) * self.w_bound, -self.w_bound, self.w_bound)

    def learn(self, z_):
        'learning'
        eta, theta = argmin(self.memory, z_, self.z_dim)
        p = 0.
        P_ = []
        Z = []
        B = []
        for i in range(len(self.memory)):
            z, w, r = self.memory[i]
            z = np.array([z])
            w = np.array(w)
            r = np.array([r])
            p = np.exp((r - np.dot(z, theta)) / eta)
            z_ = np.c_[np.array([1.]), z]
            Z.append(z_[0])
            B.append(w)
            P_.append(p[0])
        P_, B, Z = np.array(P_), np.array(B), np.array(Z)
        P = P_ * np.eye(len(self.memory))

        'calculate mean action'
        target1 = np.linalg.inv(np.dot(np.dot(Z.transpose(), P), Z))
        # print(np.shape(P), np.shape(Z.transpose()), np.shape(B))
        target2 = np.dot(np.dot(Z.transpose(), P), B)
        target = np.dot(target1, target2).transpose()
        self.a = target[:, :1]
        self.A = target[:, 1:]

        # calculate the COV
        Err = 0
        for i in range(len(self.memory)):
            z, w, r = self.memory[i]
            z = np.array([z])
            w = np.array([w])
            err = w - self.a - np.dot(self.A, z.transpose())
            Err += np.dot(err, err.transpose()) * P_[i]
        self.COV = Err / np.sum(P_) / self.COV_index
        # self.COV_index *= 1.25
        print('Upper Policy COV:', self.COV * self.COV_index)
        print('Contextual policy search upper level parameters updated')

        'record policy'
        self.Aa.append(target)
        np.save('Contextual_Policy_Search_Aa.npy', self.Aa)

    def store_data(self, z, w, r):
        transition = [z, w, [r]]
        if len(self.memory) == self.memory_dim:
            index = self.pointer % self.memory_dim  # replace the old memory with new memory
            self.memory[index] = transition
        else:
            self.memory.append(transition)
        self.pointer += 1

    def train_model(self, state_, action_):
        print('Start training forward model')
        "Parameters"
        nb_samples = 2
        input_dim = 12
        output_dim = 6
        in_idx_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        out_idx_state = [12, 13, 14, 15, 16, 17]
        in_idx_force = [0, 1, 2, 3, 4, 5]
        out_idx_force = [6, 7, 8, 9, 10, 11]
        in_idx_mean = [0]
        out_idx_mean = [1, 2, 3, 4, 5, 6]
        nb_states = 8

        'data'
        if len(state_) >= 5000:
            state_ = np.array(state_)[-5000:]
            action_ = np.array(action_)[-5000:]
        else:
            state_ = np.array(state_)
            action_ = np.array(action_)
        print('data size', np.shape(state_), np.shape(action_))

        'phase searching'
        state = []
        dpos = []
        action = []
        for i in range(len(state_) - 1):
            if state_[i + 1][0] != 0:
                if state_[i][9] >= -6.5:
                    state.append(state_[i][1:])
                    dpos.append(state_[i + 1][7:] - state_[i][7:])
                    action.append(action_[i])
        
        state = np.array(state)
        pos = state[:, 6:]
        force = state[:, :6]

        dpos = np.array(dpos)
        action = np.array(action)
        input = np.hstack([pos/self.pos_bound, action/self.action_bound])
        output = dpos/self.dpos_bound
        data_state = np.hstack([input, output])
        print(np.shape(data_state))

        'train gpr model'
        kernel = DotProduct() + WhiteKernel()
        self.gpr_force_model = GaussianProcessRegressor(kernel=kernel,
                                                        random_state=0).fit(pos/self.pos_bound, force/self.force_bound)
        self.gpr_state_model = GaussianProcessRegressor(kernel=kernel,
                                                        random_state=0).fit(input, output)

        'stage insertion'
        state = []
        dpos = []
        action = []
        for i in range(len(state_) - 1):
            if state_[i + 1][0] != 0:
                if state_[i][9] <= -5.5:
                    state.append(state_[i][1:])
                    dpos.append(state_[i + 1][7:] - state_[i][7:])
                    action.append(action_[i])
    
        state = np.array(state)
        pos = state[:, 6:]
        force = state[:, :6]
        data_force = np.hstack([pos, force]) / np.hstack([self.pos_bound, self.force_bound])

        dpos = np.array(dpos)
        action = np.array(action)
        input = np.hstack([pos/self.pos_bound, action/self.action_bound])
        output = dpos/self.dpos_bound
        data_state = np.hstack([input, output])

        self.gmr_force_model = Gmr(nb_states=nb_states, nb_dim=12, in_idx=in_idx_force, out_idx=out_idx_force)
        self.gmr_force_model.init_params_kbins(data_force.T, nb_samples=nb_samples)
        self.gmr_force_model.gmm_em(data_force.T)
        
        self.gmr_state_model = Gmr(nb_states=nb_states, nb_dim=18, in_idx=in_idx_state, out_idx=out_idx_state)
        self.gmr_state_model.init_params_kbins(data_state.T, nb_samples=nb_samples)
        self.gmr_state_model.gmm_em(data_state.T)

    def artificial_trajectory(self, State, Action, L, RL_policy, train_model):
        if train_model:
            self.train_model(State, Action)
        state = 0
        t = 0
        for i in range(len(State)):
            if State[i][0] == 0:
                state += State[i][1:]
                t += 1
        state_init = state / t
        print('initial state:', state_init)

        REWARD = []
        cal = calculation()
        l = 0
        while l < L:
            # contextual parameters
            z = np.random.randint(0, 4, 1) * 50 + 100
            cal.set_context(z)
            w = self.choose_action(z/self.z_)[0]
            cal.set_pd(w)

            # initialize the observation
            done = False
            safe = True
            R = 0
            var = np.array([0, 0, 0, 0, 0, 0, 1., 1., 0, 0, 0, 0])
            state = np.random.normal(state_init, var)
            cal.reset()

            "start a trajecory"
            for i in range(300):
                "choose action"
                # action = RL_policy.select_action(np.array(state / self.state_bound))
                action = np.array([0, 0, 0, 0, 0, 0])
                im_action = cal.step(state, action, i)

                "step"
                'calculate dpos'
                if state[8] >= -6.:
                    dpos, _ = self.gpr_state_model.predict([np.hstack([state[6:] / self.pos_bound, im_action / self.action_bound])], return_std=1, return_cov=0)
                    dpos = dpos[0]
                    dpos *= self.dpos_bound
                else:
                    dpos, cov_dpos, _ = self.gmr_state_model.gmr_predict(np.hstack([state[6:] / self.pos_bound, im_action / self.action_bound]))
                    dpos *= self.dpos_bound
                    cov_dpos = cov_dpos * self.dpos_bound ** 2 / 16
                    dpos = np.random.multivariate_normal(mean=dpos, cov=cov_dpos, size=1)[0]

                'problem check'
                if dpos[2] == 0:
                    safe = False
                if not safe:
                    print('trajectory failed')
                    break

                'calculate pos'
                pos = state[6:] + dpos

                'calculate force'
                if state[8] >= -6.:
                    force, _ = self.gpr_force_model.predict([pos/self.pos_bound], return_std=1, return_cov=0)
                    force = force[0]
                    force *= self.force_bound
                else:
                    force, cov_force, _ = self.gmr_force_model.gmr_predict(pos / self.pos_bound)
                    force *= self.force_bound
                    cov_force = cov_force * self.force_bound ** 2 / 16
                    force = np.random.multivariate_normal(mean=force, cov=cov_force, size=1)[0]

                'calculate new state'
                new_state = np.hstack([force, pos])

                reward, done, safe = cal.get_reward(state, new_state, i)

                'update parameters'
                state = new_state.copy()
                R += reward

                if done or not safe:
                    if i == 299:
                        REWARD.append([z[0], w, i, R, 'Unfinished'])
                        print('trajectory episode', l, 'step', i, w, 'reward', R, 'Unfinished')
                    elif safe == False:
                        REWARD.append([z[0], w, i, R, 'Failed'])
                        print('trajectory episode', l, 'step', i, w, 'reward', R, 'Failed')
                    else:
                        REWARD.append([z[0], w, i, R, 'Successful'])
                        print('trajectory episode', l, 'step', i, w, 'reward', R, 'Successful')
                    self.store_data(z/self.z_, w/self.w_bound, R)
                    l += 1
                    break


# **** 程序结构 ****

solver = Solver()
solver.contextual_main()
