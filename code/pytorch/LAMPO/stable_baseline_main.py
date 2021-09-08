import mujoco_py
from mujoco_py.generated import const
import pybullet as p

from envs.gym_kuka_mujoco.controllers import iMOGVIC
from envs.gym_kuka_mujoco.utils.transform_utils import *
from envs.envs_assistive.feeding_envs import *
from envs.envs_assistive.drinking_envs import *
from envs.envs_assistive.scratch_itch_envs import *

import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import time
# from .learn import make_env
# import assistive_gym

import commentjson

from code.pytorch.LAMPO.core.rl_bench_box import Mujoco_model, Mujoco_RL_model
from envs.robosuite.robosuite.controllers import *
import envs.robosuite.robosuite as suite
import numpy as np


if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

project_path = './'
sys.path.insert(0, project_path + 'code')

# ================================ stable_baseline algorithms ====================================
from stable_baselines import PPO2, SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines.common import set_global_seeds
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
# ================================================================================================

from envs.robosuite.robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from code.pytorch.LAMPO.core.rl_bench_box import *


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


def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        # module = importlib.import_module('assistive_gym.envs')
        env_class = globals()[env_name.split('-')[0] + 'Env']
        print(env_class)
        env = env_class()
    env.seed(seed)
    return env


def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()


def replay_model(env, model, deterministic=True, num_episodes=None, record=False, render=True):
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
    run_save_path = os.path.join(save_path, 'run_{}'.format(run_count))
    os.makedirs(run_save_path, exist_ok=True)
    
    params_save_path = os.path.join(run_save_path, 'params.json')
    with open(params_save_path, 'w') as f:
        commentjson.dump(params, f, sort_keys=True, indent=4, ensure_ascii=False)
    
    # Unpack options
    learning_options = params['learning_options']
    actor_options = params.get('actor_options', None)
    
    coop = 'Human' in args.env
    env = make_env(args.env, coop=True) if coop else gym.make(args.env)
 
    # log_save_path = os.path.join(save_path, 'params.json')
    # env = GymWrapper(env, logdir=save_path, max_size=learning_options["total_timesteps"])
    env = AssistiveDRL(env, params, logdir=run_save_path)
    
    # env.reset()
    # while True:
    #     # env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    #     # env.view_render()
    #     time.sleep(0.1)
    
    # Create the actor and learn
    if params['alg'] == 'PPO2':
        model = PPO(
            params['policy_type'],
            env,
            tensorboard_log=save_path,
            verbose=1,
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

    print("Learning and recording to:\n{}".format(run_save_path))
    model.learn(callback=learn_callback, **learning_options)
    model_save_path = os.path.join(run_save_path, 'model')
    model.save(model_save_path)

    return model


if __name__ == '__main__':
    import warnings
    
    parser = argparse.ArgumentParser(
        description='Runs a learning example on a registered gym environment.')
 
    parser.add_argument(
        '--param_file',
        default='VICESAssitive.json',
        type=str,
        help='the parameter file to use')
    parser.add_argument(
        '--param_dir',
        default='/home/zhimin/code/5_thu/rl-robotic-assembly-control/code/pytorch/LAMPO/params/',
        type=str,
        help='the parameter file to use')
    parser.add_argument(
        '--filter_warning',
        choices=['error', 'ignore', 'always', 'default', 'module', 'once'],
        default='default',
        help='the treatment of warnings')
    parser.add_argument(
        '--num_restarts',
        type=int,
        default=2,
        help='The number of trials to run.')
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='enables useful debug settings')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='runs in a profiler')
    
    parser.add_argument(
        '--env',
        default='DrinkingSawyerHuman-v1',
        type=str,
        help='the parameter file to use')
    
    args = parser.parse_args()
    
    # Change the warning behavior for debugging.
    warnings.simplefilter(args.filter_warning, RuntimeWarning)
    
    if args.param_file is None:
        default_path = os.path.join(args.param_dir, 'default_params.json')
        with open(default_path) as f:
            params = commentjson.load(f)[args.default_name]
    else:
        param_file = os.path.join(args.param_dir, args.param_file)
        with open(param_file) as f:
            params = commentjson.load(f)
    
    # Override some arguments in debug mode
    if args.debug or args.profile:
        params['vectorized'] = False
    
    args.env = params["env"]
    
    save_path_env_name = os.path.join('./results/assitive/', args.env)
    save_path_env_ctl = os.path.join(save_path_env_name, params["controller_options"]["impedance_mode"])
    save_path = os.path.join(save_path_env_ctl, params['alg'])
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("save_path :::", save_path)
    
    if args.profile:
        import cProfile
        for i in range(args.num_restarts):
            cProfile.run('run_learn(params, save_path, run_count=i)')
    else:
        for i in range(args.num_restarts):
            model = run_learn(args, params, save_path, run_count=i)
