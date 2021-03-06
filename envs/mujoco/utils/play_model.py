import argparse

import numpy as np

from envs.mujoco.utils.experiment_files import (get_latest_experiment_dir, get_model,
                                                get_latest_checkpoint, get_params)
from envs.mujoco.utils.load_model import load_params, load_model


# def load_params(params_path):
#     with open(params_path) as f:
#         data = json.load(f)
#     return data


# def load_model(model_path, params):
#     env_cls = globals()[params['env']]
#     orig_env = env_cls(**params['env_options'])
#     env = DummyVecEnv([lambda: orig_env])

#     if params['alg'] == 'PPO2':
#         model = PPO2.load(model_path, env=env)
#     elif params['alg'] == 'SAC':
#         model = SAC.load(model_path, env=env)
#     else:
#         raise NotImplementedError

#     return orig_env, model

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
        clipped_action = np.clip(action, env.action_space.low,
                                 env.action_space.high)
        obs, reward, done, info = env.step(clipped_action, render=render)
        if record:
            infos.append(info)
        if done:
            obs = env.reset()
            episode_count += 1

    return infos


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory', type=str, help='The directory of the experiment.')
    parser.add_argument(
        '--deterministic', action='store_true', help='Optionally simulate the deterministic system.')

    args = parser.parse_args()

    # Load the model if it's availeble, otherwise that latest checkpoint.
    experiment_dir = get_latest_experiment_dir(args.directory)
    params_path = get_params(experiment_dir)
    params = load_params(params_path)

    model_path = get_model(experiment_dir)
    if model_path is None:
        model_path = get_latest_checkpoint(experiment_dir)

    env, model = load_model(model_path, params)

    # Replay model.
    replay_model(env, model, deterministic=args.deterministic)
