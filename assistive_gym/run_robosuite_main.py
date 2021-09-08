import argparse
import os
import sys
import json
import warnings
import commentjson
import numpy as np
import gym, importlib

# from assistive_gym.wrappers.gym_wrapper import GymWrapper
from wrappers.gym_wrapper import GymWrapper

# ================================ stable_baseline algorithms ====================================
from stable_baselines3 import SAC, PPO
# from stable_baselines3.common import set_global_seeds
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.results_plotter import load_results, ts2xy
# ================================================================================================

# from envs.gym_kuka_mujoco.wrappers import TBVecEnvWrapper, TBWrapper
from assistive_gym.envs.scratch_itch_envs import ScratchItchJacoEnv

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


# def make_env(env_cls,
#              rank,
#              save_path,
#              seed=0,
#              info_keywords=None,
#              **env_options):
#
#     def _init():
#         env = env_cls(**env_options)
#         env.seed(seed + rank)
#         if info_keywords:
#             # import pdb; pdb.set_trace()
#             monitor_path = os.path.join(save_path, "proc_{}".format(rank))
#             env = Monitor(env, monitor_path, info_keywords=tuple(info_keywords))
#         return env
#
#     # set_global_seeds(seed)
#     return _init


def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        # module = importlib.import_module('assistive_gym.envs')
        # env_class = getattr(module, env_name.split('-')[0] + 'Env')
        # env = env_class()
        env = ScratchItchJacoEnv()
    env.seed(seed)
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
    '''
        Runs the learning experiment defined by the params dictionary.
        params: (dict) the parameters for the learning experiment
    '''
    # Unpack options
    learning_options = dict()
    learning_options["total_timesteps"] = args.total_timesteps
    
    actor_options = params.get('actor_options', None)
    run_save_path = os.path.join(save_path, 'run_{}'.format(run_count))
    os.makedirs(run_save_path, exist_ok=True)
    
    params_save_path = os.path.join(run_save_path, 'params.json')
    with open(params_save_path, 'w') as f:
        commentjson.dump(params, f, sort_keys=True, indent=4, ensure_ascii=False)
     
    env = make_env(args.env_name, coop=True, seed=args.seed)
    
    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    env = GymWrapper(env, run_save_path)
    
    # obs = env.reset()
    # print(obs)
    #
    # next_state = env.step(np.zeros(7))
    
    # Create the actor and learn
    if args.alg == 'PPO':
        model = PPO(
            params['policy_type'],
            env,
            verbose=10,
            tensorboard_log=run_save_path,
            **actor_options)
    elif args.alg == 'SAC':
        model = SAC(
            params['policy_type'],
            env,
            verbose=10,
            tensorboard_log=save_path,
            **actor_options
        )
    elif args.alg == 'TD3':
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
    if isinstance(model, PPO):
        learn_callback = lambda l, g: PPO_callback(l, g, run_save_path)
    elif isinstance(model, SAC):
        learn_callback = lambda l, g: SAC_callback(l, g, run_save_path)
    elif isinstance(model, TD3):
        # learn_callback = BaseCallback
        # learn_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=run_save_path)
        learn_callback = lambda l, g: TD3_callback(l, g, run_save_path)
    else:
        raise NotImplementedError
    
    # learn_callback = lambda l, g: SAC_callback(l, g, run_save_path)
    print("Learning and recording to:\n{}".format(run_save_path))
    model.learn(callback=learn_callback, **learning_options)
    model_save_path = os.path.join(run_save_path, 'model')
    model.save(model_save_path)
    
    return model


if __name__ == '__main__':
    # sys.path.append(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='Runs a learning example on a registered gym environment.')
    parser.add_argument(
        '--default_name',
        type=str,
        default='KukaMujoco-v0:PPO2', help='the name of the default entry to use')
    parser.add_argument(
        '--param_file',
        default='ImpedanceAssitance.json',
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
        default='ScratchItchJaco-v1',
        type=str,
        help='the parameter file to use')
    parser.add_argument(
            '--alg',
            default='PPO',
            type=str,
            help='the parameter file to use')
    parser.add_argument('--total_timesteps',
                        type=int,
                        default=1000000,
                        help='The number of trials to run.')
    
    parser.add_argument(
                        '--control_mode',
                        default='variable',
                        type=str,
                        help='the parameter file to use')
    parser.add_argument(
                        '--robot_name',
                        default="Panda",
                        type=str,
                        help='the parameter file to use')
    parser.add_argument(
                        '--seed',
                        default=1,
                        type=int,
                        help='the parameter file to use')
    parser.add_argument(
        "--render",
        default=False)
    
    args = parser.parse_args()
    
    # Change the warning behavior for debugging.
    warnings.simplefilter(args.filter_warning, RuntimeWarning)
    
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
    
    # Override some arguments in debug mode
    if args.debug or args.profile:
        params['vectorized'] = False
    
    save_path_env_name = os.path.join('./results/ral-2021/', args.env_name)
    save_path_env_ctl = os.path.join(save_path_env_name, args.control_mode)
    save_path = os.path.join(save_path_env_ctl, params['alg'])
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("save_path :::", save_path)
    
    if args.profile:
        import cProfile
        for i in range(args.num_restarts):
            cProfile.run('run_learn(params, save_path, run_count=i)')
    else:
        # for i in range(args.num_restarts):
        model = run_learn(args, params, save_path, run_count=args.seed)
