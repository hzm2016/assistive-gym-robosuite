import argparse
import os
import sys
import json
import commentjson

project_path = './'
sys.path.insert(0, project_path + 'code')
import envs.robosuite.robosuite as suite
import numpy as np

# ================================ stable_baseline algorithms ====================================
from stable_baselines import PPO2, SAC
from stable_baselines.common import set_global_seeds
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

# ================================================================================================

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

import warnings

from code.pytorch.PLAS import utils
from code.pytorch.PLAS import algos
from code.pytorch.PLAS.logger import logger, setup_logger
import d4rl
import torch
import time
from code.pytorch.PLAS.eval_functions import eval_critic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    info = {'AverageReturn': avg_reward}
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return info


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))
                
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)


def TD3_callback(_locals, _globals, log_dir):
    """
        Callback called at each gradient update.
    """
    # Get the current update step.
    # n_update = _locals['n_update']
    #
    # # Save on the first update and every 10 updates after that.
    # if (n_update == 1) or (n_update % 10 == 0):
    # 	checkpoint_save_path = os.path.join(
    # 		log_dir, 'model_checkpoint_{}.pkl'.format(n_update))
    # 	_locals['self'].save(checkpoint_save_path)
    pass


def PPO_callback(_locals, _globals, log_dir):
    """
        Callback called at each gradient update.
    """
    # Get the current update step.
    # n_update = _locals['n_update']
    #
    # # Save on the first update and every 10 updates after that.
    # if (n_update == 1) or (n_update % 10 == 0):
    # 	checkpoint_save_path = os.path.join(
    # 		log_dir, 'model_checkpoint_{}.pkl'.format(n_update))
    # 	_locals['self'].save(checkpoint_save_path)
    pass


PPO_callback.n_update = 0


def SAC_callback(_locals, _globals, log_dir):
    """
        Callback called at each gradient update.
    """
    # new_update = SAC_callback.n_updates < _locals['n_updates']
    # if new_update:
    # 	SAC_callback.n_updates = _locals['n_updates']
    #
    # # Save on the first update and every 10 updates after that.
    # if new_update and ((SAC_callback.n_updates == 1) or
    # 				   (SAC_callback.n_updates % 1000 == 0)):
    # 	checkpoint_save_path = os.path.join(
    # 		log_dir, 'model_checkpoint_{}.pkl'.format(SAC_callback.n_updates))
    # 	_locals['self'].save(checkpoint_save_path)
    pass


SAC_callback.n_updates = 0


def make_env(args, params, save_path='', run_count=1):
    """
        Utility function for vectorized env.
    """
    
    # Unpack options
    learning_options = params['learning_options']
    actor_options = params.get('actor_options', None)
    algorithm_kwargs = params.get('algorithm_kwargs')
    eval_environment_kwargs = params.get('eval_environment_kwargs')
    
    run_save_path = os.path.join(save_path, 'run_{}'.format(run_count))
    os.makedirs(run_save_path, exist_ok=True)
    
    # # Save the parameters that will generate the model
    params_save_path = os.path.join(run_save_path, 'params.json')
    with open(params_save_path, 'w') as f:
        commentjson.dump(params, f, sort_keys=True, indent=4, ensure_ascii=False)
    
    controller_name = "OSC_POSE"
    np.random.seed(3)
    
    # Define controller path to load
    controller_path = os.path.join(os.path.dirname(__file__),
                                   'robosuite',
                                   'controllers/config/{}.json'.format(controller_name.lower()))
    
    # Load the controller
    with open(controller_path) as f:
        controller_config = json.load(f)
    
    # Manually edit impedance settings
    controller_config["impedance_mode"] = args.control_mode
    controller_config["kp_limits"] = [10, 300]
    controller_config["damping_limits"] = [0, 10]
    
    # Now, create a test env for testing the controller on
    env = suite.make(
        args.env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=eval_environment_kwargs.get('horizon'),
        control_freq=eval_environment_kwargs.get('control_freq'),
        controller_configs=controller_config
    )
    
    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    # if args.render:
    #     env.viewer.set_camera(camera_id=0)
    
    from envs.robosuite.robosuite.wrappers.gym_wrapper import GymWrapper
    env = GymWrapper(env, logdir=save_path, max_size=learning_options["total_timesteps"])
    
    return env


def replay_model(env, model, deterministic=True, num_episodes=None, record=False, render=True):
    # Don't record data forever.
    assert (not record) or (num_episodes is not None), \
        "there must be a finite number of episodes to record the data"
    
    # Initialize counts and data.
    num_episodes = num_episodes if num_episodes else np.inf
    episode_count = 0
    infos = []
    
    # Simulate forward.
    obs = env.reset()
    while episode_count < num_episodes:
        # import pdb; pdb.set_trace()
        action, _states = model.predict(obs, deterministic=deterministic)
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, info = env.step(clipped_action, render=render)
        if record:
            infos.append(info)
        
        if done:
            obs = env.reset()
            episode_count += 1
    
    return infos


def run_learn(args, params, save_path='', run_count=1):
    """
        Runs the learning experiment defined by the params dictionary.
        params: (dict) the parameters for the learning experiment
    """
    # Unpack options
    learning_options = params['learning_options']
    actor_options = params.get('actor_options', None)
    algorithm_kwargs = params.get('algorithm_kwargs')
    eval_environment_kwargs = params.get('eval_environment_kwargs')
    
    run_save_path = os.path.join(save_path, 'run_{}'.format(run_count))
    os.makedirs(run_save_path, exist_ok=True)
    
    # # Save the parameters that will generate the model
    params_save_path = os.path.join(run_save_path, 'params.json')
    with open(params_save_path, 'w') as f:
        commentjson.dump(params, f, sort_keys=True, indent=4, ensure_ascii=False)
    
    controller_name = "OSC_POSE"
    np.random.seed(3)
    
    # Define controller path to load
    controller_path = os.path.join(os.path.dirname(__file__),
                                   'robosuite',
                                   'controllers/config/{}.json'.format(controller_name.lower()))
    
    # Load the controller
    with open(controller_path) as f:
        controller_config = json.load(f)
    
    # Manually edit impedance settings
    controller_config["impedance_mode"] = args.control_mode
    controller_config["kp_limits"] = [10, 300]
    controller_config["damping_limits"] = [0, 10]
    
    # Now, create a test env for testing the controller on
    env = suite.make(
        args.env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=eval_environment_kwargs.get('horizon'),
        control_freq=eval_environment_kwargs.get('control_freq'),
        controller_configs=controller_config
    )
    
    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    # if args.render:
    #     env.viewer.set_camera(camera_id=0)
    
    from envs.robosuite.robosuite.wrappers.gym_wrapper import GymWrapper
    env = GymWrapper(env, logdir=save_path, max_size=learning_options["total_timesteps"])
    
    # env.reset()
    #
    # env.model.save_model('panda.xml')
    #
    # while True:
    #     env.render()
    
    # envs = [
    # 	make_env(
    # 		env_cls,
    # 		i,
    # 		run_save_path,
    # 		info_keywords=params.get('info_keywords', None),
    # 		**params['env_options']) for i in range(params['n_env'])
    # ]
    # # envs = [make_env(params['env'], i, save_path) for i in range(params['n_env'])]
    
    # if params.get('vectorized', True):
    # 	env = SubprocVecEnv(envs)
    # else:
    # 	env = DummyVecEnv(envs)
    
    # env = DummyVecEnv(envs)
    #
    # print('info_keywords :::', params.get('info_keywords', tuple()))
    # env = TBVecEnvWrapper(
    # 	env, save_path, info_keywords=params.get('info_keywords', tuple()))
    
    # env = TBWrapper(
    # 	env, save_path, info_keywords=params.get('info_keywords', tuple()))
    
    # Create the actor and learn
    if params['alg'] == 'PPO2':
        model = PPO2(
            params['policy_type'],
            env,
            tensorboard_log=save_path,
            **actor_options)
    elif params['alg'] == 'SAC':
        model = SAC(
            params['policy_type'],
            env,
            tensorboard_log=save_path,
            **actor_options)
    elif params['alg'] == 'TD3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3(
            params['policy_type'],
            env,
            action_noise=action_noise,
            # verbose=1,
            tensorboard_log=save_path,
            **actor_options)
    else:
        raise NotImplementedError
    
    # Create the callback
    if isinstance(model, PPO2):
        learn_callback = lambda l, g: PPO_callback(l, g, run_save_path)
    elif isinstance(model, SAC):
        learn_callback = lambda l, g: SAC_callback(l, g, run_save_path)
    elif isinstance(model, TD3):
        # learn_callback = BaseCallback
        # learn_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=run_save_path)
        learn_callback = lambda l, g: TD3_callback(l, g, run_save_path)
    else:
        raise NotImplementedError
    
    print("Learning and recording to:\n{}".format(run_save_path))
    model.learn(callback=learn_callback, **learning_options)
    model_save_path = os.path.join(run_save_path, 'model')
    model.save(model_save_path)
    
    env.save_buffer()
    
    return model


if __name__ == '__main__':
    # Setup command line arguments.
    parser = argparse.ArgumentParser(
        description='Runs a learning example on a registered gym environment.')
    parser.add_argument(
        '--default_name',
        type=str,
        default='KukaMujoco-v0:PPO2', help='the name of the default entry to use')
    parser.add_argument(
        '--param_file',
        default='params/ImpedanceV2PegInsertion:SAC.json',
        # default='manipulation/pushing/ImpedanceV2Pushing:PPO2.json',
        # default='manipulation/peg_insertion/ImpedanceV2PegInsertion:TD3.json',
        type=str,
        help='the parameter file to use')
    parser.add_argument(
        '--filter_warning',
        choices=['error', 'ignore', 'always', 'default', 'module', 'once'],
        default='default',
        help='the treatment of warnings')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='enables useful debug settings')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='runs in a profiler')
    parser.add_argument(
        '--final',
        action='store_true',
        help='puts the data in the final directory for easy tracking/plotting')
    parser.add_argument(
        '--num_restarts',
        type=int,
        default=1,
        help='The number of trials to run.')
    
    parser.add_argument(
        '--env_name',
        default='Wipe',
        type=str,
        help='the parameter file to use')
    parser.add_argument(
        '--control_mode',
        default='variable_kp',
        type=str,
        help='the parameter file to use')
    
    parser.add_argument("--render",
                        default=False)
    
    # Additional parameters
    parser.add_argument("--ExpID",
                        default=9999,
                        type=int)  # Experiment ID
    parser.add_argument('--log_dir',
                        default='./results/', type=str)  # Logging directory
    parser.add_argument("--load_model",
                        default=None, type=str)  # Load model and optimizer parameters
    parser.add_argument("--save_model",
                        default=True, type=bool)  # Save model and optimizer parameters
    parser.add_argument("--save_freq",
                        default=1e5, type=int)  # How often it saves the model

    parser.add_argument("--algo_name",
                        default="Latent")  # Algorithm: Latent or LatentPerturbation.
    parser.add_argument("--dataset",
                        default=None, type=str)  # path to dataset if not d4rl env
    parser.add_argument("--seed",
                        default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq",
                        default=1e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps",
                        default=5e5, type=int)  # Max time steps to run environment for
    parser.add_argument('--vae_mode',
                        default='train', type=str)  # VAE mode: train or load from a specific version
    parser.add_argument('--vae_lr',
                        default=1e-4, type=float)  # vae training iterations
    parser.add_argument('--vae_itr',
                        default=500000,
                        type=int)  # vae training iterations
    parser.add_argument('--vae_hidden_size',
                        default=750,
                        type=int)  # vae training iterations
    parser.add_argument('--max_latent_action',
                        default=2.,
                        type=float)  # max action of the latent policy
    parser.add_argument('--phi', default=0., type=float)  # max perturbation
    parser.add_argument('--batch_size', default=100, type=int)  # batch size
    parser.add_argument('--actor_lr', default=1e-4, type=float)  # policy learning rate
    parser.add_argument('--critic_lr', default=1e-3, type=float)  # policy learning rate
    parser.add_argument('--tau', default=0.005, type=float)  # actor network size
    
    args = parser.parse_args()
    
    # Change the warning behavior for debugging.
    warnings.simplefilter(args.filter_warning, RuntimeWarning)
    
    from code.pytorch.PLAS.utils import *
    
    # Setup Logging
    file_name = f"Exp{args.ExpID:04d}_{args.algo_name}_{args.dataset}-{args.seed}"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print(folder_name)
    
    variant = vars(args)
    variant.update(node=os.uname()[1])
    setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)
    
    # Setup Environment
    # Load the learning parameters from a file.
    param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'params')

    if args.param_file is None:
        default_path = os.path.join(param_dir, 'default_params.json')
        with open(default_path) as f:
            params = commentjson.load(f)[args.default_name]
    else:
        param_file = os.path.join(param_dir, args.param_file)
        with open(param_file) as f:
            params = commentjson.load(f)
    
    save_path_env_name = os.path.join('./results/com_offline/', args.env_name)
    save_path_env_ctl = os.path.join(save_path_env_name, args.control_mode)
    save_path = os.path.join(save_path_env_ctl, params['alg'])
    
    env = make_env(args, params, save_path, run_count=0)
    
    state_dim = env.observation_space.shape[0]
    print("state_dim :", state_dim)
    action_dim = env.action_space.shape[0]
    print("action_dim", action_dim)
    max_action = float(env.action_space.high[0])
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e6))
    buffer_data = np.load('./results/com_tmech/Wipe/variable_kp/SAC/saved_buffer.npy', allow_pickle=True).item()
    replay_buffer.load_buffer(buffer_data)
    print("replay_buffer :", replay_buffer)
    
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Train or Load VAE
    latent_dim = action_dim * 2
    vae_trainer = algos.VAEModule(state_dim, action_dim, latent_dim, max_action,
                                  vae_lr=args.vae_lr, hidden_size=args.vae_hidden_size)
    
    if args.vae_mode == 'train':
        # Train VAE
        print(time.ctime(), "Training VAE...")
        logs = vae_trainer.train(replay_buffer.storage, folder_name, iterations=args.vae_itr)
    else:
        # Select vae automatically
        vae_dirname = os.path.dirname(os.path.abspath(__file__)) + '/models/vae_' + args.vae_mode
        vae_filename = args.dataset + '-' + str(args.seed)
        vae_trainer.load(vae_filename, vae_dirname)
        print('Loaded VAE from:' + os.path.join(vae_dirname, vae_filename))

    policy = None
    if args.algo_name == 'Latent':
        policy = algos.Latent(vae_trainer.vae, state_dim, action_dim, latent_dim, max_action, **vars(args))
    elif args.algo_name == 'LatentPerturbation':
        policy = algos.LatentPerturbation(vae_trainer.vae, state_dim, action_dim, latent_dim, max_action, **vars(args))

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0
    while training_iters < args.max_timesteps:
    
        # Train
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        training_iters += args.eval_freq
        print("Training iterations: " + str(training_iters))
        logger.record_tabular('Training Epochs', int(training_iters // int(args.eval_freq)))
    
        # Save Model
        if training_iters % args.save_freq == 0 and args.save_model:
            policy.save('model_' + str(training_iters), folder_name)
    
        # Eval
        info = eval_policy(policy, env)
        evaluations.append(info['AverageReturn'])
        np.save(os.path.join(folder_name, 'eval'), evaluations)
        
        # eval_dict = eval_critic(policy.select_action, policy.critic.q1, env)
        # for k, v in eval_dict.items():
        #     logger.record_tabular('Eval_critic/' + k, v)
    
        for k, v in info.items():
            logger.record_tabular(k, v)
    
        logger.dump_tabular()

    # # Override some arguments in debug mode
    # if args.debug or args.profile:
    # 	params['vectorized'] = False
    #
    # print("params :::", params)
    #
    # save_path_env_name = os.path.join('./results/com_tmech/', args.env_name)
    # save_path_env_ctl = os.path.join(save_path_env_name, args.control_mode)
    # save_path = os.path.join(save_path_env_ctl, params['alg'])
    #
    # if not os.path.exists(save_path):
    # 	os.makedirs(save_path)
    #
    # print("save_path :::", save_path)
    #
    # if args.profile:
    # 	import cProfile
    #
    # 	for i in range(args.num_restarts):
    # 		cProfile.run('run_learn(params, save_path, run_count=i)')
    # else:
    # 	for i in range(args.num_restarts):
    # 		model = run_learn(args, params, save_path, run_count=i)
    
    # print(dataset['state'].shape[0])
    # for i in range(dataset['states'].shape[0] - 1):
    # 	self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
    # 	         data['rewards'][i], data['terminals'][i])
    # print("Dataset size:" + str(self.size))
