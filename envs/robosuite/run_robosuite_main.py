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


def make_env(env_cls,
             rank,
             save_path,
             seed=0,
             info_keywords=None,
             **env_options):
    """
        Utility function for vectorized env.
    
        :param env_cls: (str) the environment class to instantiate
        :param rank: (int) index of the subprocess
        :param seed: (int) the inital seed for RNG
        :param info_keywords: (tuple) optional, the keywords to record
        :param **env_options: additional arguments to pass to the environment
    """
    
    def _init():
        env = env_cls(**env_options)
        env.seed(seed + rank)
        if info_keywords:
            # import pdb; pdb.set_trace()
            monitor_path = os.path.join(save_path, "proc_{}".format(rank))
            env = Monitor(env, monitor_path, info_keywords=tuple(info_keywords))
        return env
    
    set_global_seeds(seed)
    return _init


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
    controller_config["impedance_mode"] = "fixed"
    controller_config["kp_limits"] = [10, 300]
    controller_config["damping_limits"] = [0, 10]
    
    # Now, create a test env for testing the controller on
    env = suite.make(
        args.env_name,
        robots="Jaco",
        has_renderer=True,
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
    
    env.reset()
    
    # env.model.save_model('Jaco.xml')
    
    while True:
        env.step(np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]))
        env.render()
    
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

    # print("Learning and recording to:\n{}".format(run_save_path))
    # model.learn(callback=learn_callback, **learning_options)
    # model_save_path = os.path.join(run_save_path, 'model')
    # model.save(model_save_path)
    #
    # env.save_buffer()

    return model


if __name__ == '__main__':
    import warnings
    
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
        default='variable',
        type=str,
        help='the parameter file to use')
    
    parser.add_argument("--render",
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
    
    print("params :::", params)
    
    save_path_env_name = os.path.join('./results/com_tmech/', args.env_name)
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
        for i in range(args.num_restarts):
            model = run_learn(args, params, save_path, run_count=i)

    # model = run_learn(params, save_path, run_count=i)
    # run_learn(args)
