from code.pytorch.LAMPO.core.config import config
from code.pytorch.LAMPO.core.collector import RunModel, RunModelMujoco, RunModelPybullet
from code.pytorch.LAMPO.core.colome_torras import CT_ImitationLearning, CT_ReinforcementLearning
from code.pytorch.LAMPO.core.task_interface import TaskInterface
from code.pytorch.LAMPO.core.plot import LampoMonitor
from code.pytorch.LAMPO.core.lampo import Lampo

from mppca.mixture_ppca import MPPCA
from envs.gym_kuka_mujoco.envs import *
from code.pytorch.LAMPO.core.rl_bench_box import *

from mppca.mixture_ppca import MPPCA
from code.pytorch.LAMPO.core.lampo import Lampo
from code.pytorch.LAMPO.core.rl_bench_box import *
from code.pytorch.LAMPO.core.model import RLModel

# two pybullet environments
import gym
from envs.envs_assistive.feeding_envs import *
from envs.envs_assistive.drinking_envs import *

import argparse
import numpy as np
np.set_printoptions(precision=5)
import os
import json
import commentjson
import importlib

from tensorboardX import SummaryWriter as FileWriter


def viewer(env_name):
    coop = 'Human' in env_name
    # env = make_env(env_name, coop=True) if coop else gym.make(env_name)
    
    env = FeedingSawyerHumanEnv()
    # env = DrinkingSawyerHumanEnv()
    
    while True:
        done = False
        env.render()
        observation = env.reset()
        print("observation :", observation)
        print("target pos :", env.target_pos)
        
        # print("target_ori :", env.target_orient)
        # print("config :::",
        #       os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config.ini'))
        # time.sleep(1)
        
        action = sample_action(env, coop)
        print("action ::::", action)
        
        if coop:
            print('Robot observation size:', np.shape(observation['robot']),
                  'Human observation size:', np.shape(observation['human']),
                  'Robot action size:', np.shape(action['robot']),
                  'Human action size:', np.shape(action['human'])
                  )
        else:
            print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))
        
        while not done:
            action = sample_action(env, coop)
            observation, reward, done, info = env.step(action)
            
            if coop:
                done = done['__all__']
            print('Robot reward:', reward['robot'], 'Human reward:', reward['human'])


class Objectview(object):
    
    def __init__(self, d):
        self.__dict__ = d
        
        
def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        # module = importlib.import_module('assistive_gym.envs')
        env_class = globals()[env_name.split('-')[0] + 'Env']
        env = env_class()
    env.seed(seed)
    return env


def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()


def process_parameters(parameters, n_samples, n_context, noise=0.03):
    parameters = parameters[:n_samples].copy()
    parameters[:, :n_context] += noise * np.random.normal(size=parameters[:, :n_context].shape)
    return parameters


def train(args, task, parameters, params):
    """
        process parameters
    """
    n_clusters = params["alg_options"]["n_cluster"]
    state_dim = params["alg_options"]["state_dim"]
    latent_dim = int(params["alg_options"]["latent_dim"])
    action_dim = params["alg_options"]["action_dim"]
    n_features = params["alg_options"]["n_features"]
    
    parameters = process_parameters(parameters,
                                    args.imitation_learning,
                                    state_dim,
                                    args.il_noise)
    print("parameters :::", parameters.shape)
    print("n_cluster :::", n_clusters)
    print("latent_dim :::", latent_dim)
    
    n_evaluation_samples = args.n_evaluations
    n_batch = args.batch_size
    
    kl_bound = args.kl_bound
    kl_context_bound = args.context_kl_bound
    if args.forward:
        kl_type = "forward"
    else:
        kl_type = "reverse"
    
    if args.alg == 'REPS':
        imitation = CT_ImitationLearning(state_dim,
                                         params["alg_options"]["parameter_dim"],
                                         params["alg_options"]["latent_dim"],
                                         n_clusters, use_dr=args.use_dr)
        imitation.fit(parameters[:, :state_dim],
                      parameters[:, state_dim:],
                      forgetting_rate=args.forgetting_rate)
        
        rl_model = CT_ReinforcementLearning(imitation, kl_bound=kl_bound)
    elif args.alg == 'MPPCA':
        print("MPPCA training :::")
        mppca = MPPCA(n_clusters, latent_dim, n_init=200)
        mppca.fit(parameters)
        
        rl_model = RLModel(mppca,
                           context_dim=state_dim,
                           kl_bound=kl_bound,
                           kl_bound_context=kl_context_bound,
                           kl_reg=args.context_reg,
                           normalize=args.normalize,
                           kl_type=kl_type
                           )
        
        sr = Lampo(rl_model, wait=not args.slurm)
        
        # myplot = LampoMonitor(kl_bound, kl_context_bound=0.,
        #                       title="class_log kl=%.2f, %d samples" %
        #                             (kl_bound, n_batch))
        #
        # sr = Lampo(rl_model, wait=not args.slurm)
    else:
        print("Please give parameter fit model !")

    collector = RunModelPybullet(task, rl_model, args.dense_reward)

    reward_list = []
    success_list = []
    context_list = []
    parameter_list = []
    if not os.path.exists("results/assitive/" + args.flag + '/'):
        os.makedirs("results/assitive/" + args.flag + '/')
    logdir = "results/assitive/" + args.flag + '/' + args.env + '_' + args.alg + '/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    reward_writer = FileWriter(logdir)
    for i in range(args.max_iter):
        reward_episodes, success_episodes, actions, latent, cluster, observation = \
            collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)
        context_list.append(observation)
        parameter_list.append(actions)
        
        if args.alg == 'REPS':
            rl_model.add_dataset(observation[:n_batch], actions[:n_batch], reward_episodes[:n_batch])
        
        if args.alg == 'MPPCA':
            sr.add_dataset(actions[:n_batch], latent[:n_batch], cluster[:n_batch],
                           observation[:n_batch], reward_episodes[:n_batch])
            sr.improve()
        
        reward_writer.add_scalar('episode_reward', np.mean(reward_episodes), int(i))
        reward_writer.add_scalar('success_rate', np.sum(success_episodes) / n_evaluation_samples, int(i))
        
        reward_list.append(np.mean(reward_episodes).copy())
        success_list.append(np.sum(success_episodes).copy()/n_evaluation_samples)
        
        print("ITERATION :", i)
        print("Reward_list :", np.mean(reward_episodes).copy())
        print("SUCCESS :", np.sum(success_episodes).copy()/n_evaluation_samples)

        np.save("results/assitive/" + args.flag + '/' + args.env + '_' + args.alg + "_reward_list.npy",
                np.array(reward_list))
        np.save("results/assitive/" + args.flag + '/' + args.env + '_' + args.alg + "_success_list.npy",
                np.array(success_list))
    
    # s, r, actions, latent, cluster, observation = \
    #     collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)

    np.save("results/assitive/" + args.flag + '/' + args.env + '_' + args.alg + "_reward_list.npy",
            np.array(reward_list))
    np.save("results/assitive/" + args.flag + '/' + args.env + '_' + args.alg + "_success_list.npy",
            np.array(success_list))
    
    np.save("results/assitive/" + args.flag + '/' + args.env + '_' + args.alg + "_context_list.npy",
            np.array(context_list))
    np.save("results/assitive/" + args.flag + '/' + args.env + '_' + args.alg + "_parameter_list.npy",
            np.array(parameter_list))
    print("-" * 50)
    print("ITERATION", args.max_iter)
    print("-" * 50)


def eval(args, task, params):
    task.reset()
    print("context :", task.read_context())
    task.set_waypoints(way_points_list=None)
    
    task.send_movement(params=None)
    parameters, reward_list = task.get_demonstrations(num_traj=10)
    print("reward_list :", reward_list)
    
    while True:
        done = False
        task._env.reset()
        task._env.render()
        time.sleep(2)

    # weights = np.ones(n_features * action_dim)
    # task.send_movement(weights, 5)
    # print("trajectory :", task.send_movement(weights, 5).values.shape)
    

def get_arguments_dict():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t",
                        "--task_name",
                        help="Task name.",
                        default="reach_target")
    parser.add_argument("-i",
                        "--id",
                        help="Identifier of the process.",
                        type=int, default=10)
    parser.add_argument(
                        "--batch_size",
                        help="How many episodes before improvement.",
                        type=int,
                        default=50)
    parser.add_argument(
                        "--imitation_learning",
                        help="How many episodes before improvement.",
                        type=int,
                        default=200)
    
    parser.add_argument("--il_noise",
                        help="Add noise on the context",
                        type=float,
                        default=0.03)
    parser.add_argument("--dense_reward",
                        help="Use dense reward",
                        default=False)
    parser.add_argument("-k", "--kl_bound",
                        help="Bound the improvement kl.",
                        type=float,
                        default=0.2)
    parser.add_argument("-c", "--context_kl_bound",
                        help="Bound the context kl.",
                        type=float,
                        default=50.)
    parser.add_argument("--context_reg",
                        help="Bound the improvement kl.",
                        type=float,
                        default=1E-4)
    parser.add_argument("-z",
                        "--normalize",
                        help="Normalized Importance Sampling",
                        default=True)
    parser.add_argument("-m",
                        "--max_iter",
                        help="Maximum number of iterations.",
                        type=int,
                        default=15)
    parser.add_argument("--n_evaluations",
                        help="Number of the evaluation batch.",
                        type=int,
                        default=200)
    parser.add_argument("--use_dr",
                        help="Don't do dimensionality reduction.",
                        default=True)
    parser.add_argument("--plot",
                        help="Don't do dimensionality reduction.",
                        default=True)
    parser.add_argument("--forgetting_rate",
                        help="The forgetting rate of the IRWR-GMM.",
                        type=float,
                        default=1.)

    parser.add_argument('--video_record',
                        type=bool,
                        default=False,
                        help='the parameter file to use')

    parser.add_argument('--video_path',
                        type=str,
                        default='IMOGICAssitive.json',
                        help='the parameter file to use')
    
    parser.add_argument('--param_dir',
                        type=str,
                        default='params/',
                        help='the parameter file to use')
    parser.add_argument('--param_file',
                        type=str,
                        default='IMOGICAssitive.json',
                        help='the parameter file to use')
    
    parser.add_argument('--filter_warning',
                        choices=['error', 'ignore', 'always', 'default', 'module', 'once'],
                        default='default',
                        help='the treatment of warnings')
    parser.add_argument('--profile',
                        action='store_true',
                        help='runs in a profiler')
    parser.add_argument('--final',
                        action='store_true',
                        help='puts the data in the final directory for easy tracking/plotting')
    
    parser.add_argument("--env",
                        default="DrinkingSawyerHuman-v1",
                        # default="FeedingSawyerHuman-v1"
                        )
    
    parser.add_argument("-r",
                        "--slurm",
                        help="Don't look for CPU usage.",
                        action="store_true")
    parser.add_argument("--forward",
                        help="Bound the improvement kl.",
                        default=False)
    
    parser.add_argument("--alg",
                        default='REPS')
    parser.add_argument("--flag",
                        default='multi_waypoints')
    
    parser.add_argument("--load_data",
                        default=False)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments_dict()

    current_path = os.path.abspath(__file__)
    root_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    param_dir = os.path.join(root_path, args.param_dir)
    param_file = os.path.join(param_dir, args.param_file)

    with open(param_file) as f:
        params = commentjson.load(f)
    
    args.env = params["env"]
    
    coop = 'Human' in args.env
    env = make_env(args.env, coop=True) if coop else gym.make(args.env)
    
    task = globals()[params["alg_options"]["task_class"]](args, env, params)
    print("context_dim :", task.get_context_dim())
    print("latent_dim :", task.get_impedance_dim())

    ###################################################################
    ###################################################################
    save_path = 'results/demo/multi_waypoints/'
    print('+' * 100)
    if args.load_data:
        print('Load datasets !')
        parameters = np.load(save_path + args.env + '_demonstration.npy')
    else:
        print('Collect datasets !')
        parameters, reward_list, success_list = task.get_demonstrations(num_traj=args.imitation_learning)
        # parameters = task.get_demonstrations(num_traj=50)[:args.imitation_learning]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(save_path + args.env + '_' + 'demonstration.npy', parameters)
        np.save(save_path + args.env + '_' + 'reward_list.npy', reward_list)
        np.save(save_path + args.env + '_' + 'success_list.npy', success_list)

    print("parameters :::", parameters)
    train(args, task, parameters, params)

