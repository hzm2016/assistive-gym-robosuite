from code.pytorch.LAMPO.core.config import config
import argparse
from code.pytorch.LAMPO.core.collector import RunModel, RunModelMujoco
from code.pytorch.LAMPO.core.colome_torras import CT_ImitationLearning, CT_ReinforcementLearning
from code.pytorch.LAMPO.core.task_interface import TaskInterface
from code.pytorch.LAMPO.core.plot import LampoMonitor
import numpy as np
import json
import os
import commentjson

from mppca.mixture_ppca import MPPCA
from code.pytorch.LAMPO.core.lampo import Lampo
from code.pytorch.LAMPO.core.rl_bench_box import *
from code.pytorch.LAMPO.core.model import RLModel

from envs.gym_kuka_mujoco.envs import *


def get_arguments_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task_name",
                        help="Task name.",
                        default="reach_target")
    parser.add_argument("-i", "--id",
                        help="Identifier of the process.",
                        type=int, default=10)
    parser.add_argument(
                        "--batch_size",
                        help="How many episodes before improvement.",
                        type=int,
                        default=30)
    parser.add_argument(
                        "--imitation_learning",
                        help="How many episodes before improvement.",
                        type=int, default=50)
    
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
    parser.add_argument("-z", "--normalize",
                        help="Normalized Importance Sampling",
                        default=True)
    parser.add_argument("-m",
                        "--max_iter",
                        help="Maximum number of iterations.",
                        type=int,
                        default=1)
    parser.add_argument("--n_evaluations",
                        help="Number of the evaluation batch.",
                        type=int,
                        default=20)
    parser.add_argument("--not_dr",
                        help="Don't do dimensionality reduction.",
                        default=False)
    parser.add_argument("--plot",
                        help="Don't do dimensionality reduction.",
                        default=True)
    parser.add_argument("--forgetting_rate",
                        help="The forgetting rate of the IRWR-GMM.",
                        type=float,
                        default=1.)
    parser.add_argument('--param_file',
                        type=str, default='h-cps/assisitive/IMOGICPegInsertion.json',
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
    parser.add_argument('--num_restarts',
                        type=int,
                        default=1,
                        help='The number of trials to run.')
    
    parser.add_argument("--policy_name",
                        default='SAC')
    parser.add_argument("--env_name",
                        default="insertion")  # OpenAI gym environment name
    parser.add_argument("--log_path",
                        default='com_cps/vic')
    parser.add_argument("--eval_only",
                        default=False)
    parser.add_argument("--eval_max_timesteps",
                        default=1e3,
                        type=int)

    # ============================= Policy Search : GPREPS ===================================
    parser.add_argument("--num_policy_update",
                        default=1000,
                        type=int)
    parser.add_argument("--num_real_episodes",
                        default=5,
                        type=int)
    parser.add_argument("--num_simulated_episodes",
                        default=15,
                        type=int)
    parser.add_argument("--num_average_episodes",
                        default=10,
                        type=int)
    parser.add_argument("--max_episode_steps",
                        default=400,
                        type=int)
    parser.add_argument("--start_policy_update_idx",
                        default=25,
                        type=int)
    parser.add_argument("--eval_policy_update",
                        default=10,
                        type=int)
    parser.add_argument("--eval_max_context_pairs",
                        default=5,
                        type=int)
    parser.add_argument("--max_eval_episode",
                        default=5,
                        type=int)
    parser.add_argument("--render",
                        default=False)
    parser.add_argument("--eps",
                        default=[0.2, 0.65])
    parser.add_argument("-r", "--slurm",
                        help="Don't look for CPU usage.",
                        action="store_true")
    parser.add_argument("--forward",
                        help="Bound the improvement kl.",
                        default=False)

    parser.add_argument("--alg",
                        default='MPPCA')

    parser.add_argument("--load_data",
                        default=True)
    
    args = parser.parse_args()
    return args


class Objectview(object):
 
    def __init__(self, d):
        self.__dict__ = d


def process_parameters(parameters, n_samples, n_context, noise=0.03):
    parameters = parameters[:n_samples].copy()
    parameters[:, :n_context] += noise * np.random.normal(size=parameters[:, :n_context].shape)
    return parameters


if __name__ == "__main__":
    args = get_arguments_dict()

    param_dir = 'params/'
    param_file = 'IMOGICPegInsertion.json'
    
    param_file = os.path.join(param_dir, param_file)
    with open(param_file) as f:
        set_params = commentjson.load(f)
    
    # make env
    env_cls = globals()[set_params['env']]
    env = env_cls(**set_params['env_options'])
    
    n_clusters = set_params["alg_options"]["n_cluster"]
    
    task = globals()[set_params["alg_options"]["task_class"]](env, set_params)
    print("context_dim :", task.get_context_dim())
    print("latent_dim :", task.get_impedance_dim())
    # print("context :", task.read_context())
    print("+" * 100)

    state_dim = task.get_context_dim()
    # print("state_dim :", state_dim)
    #
    # # reward, info = task.send_movement(params)
    # # print("reward :", reward, "info", info)
    
    print('+' * 100)
    if args.load_data:
        print('Load datasets !')
        # parameters = np.load('./demo/insertion_demonstration.npy')
        env_name = 'FeedingSawyerHuman-v1'
        parameters = np.load('./demo/' + env_name + '_demonstration.npy')
    else:
        print('Collect datasets !')
        parameters = task.get_demonstrations(num_traj=50)[:args.imitation_learning]
        # print("parameters :::", parameters[:, state_dim:])
        np.save('./demo/insertion_demonstration.npy', parameters)

    print("parameters :", parameters.shape)
    parameters = process_parameters(parameters,
                                    args.imitation_learning,
                                    state_dim,
                                    args.il_noise)
    # print("parameters :::", parameters.shape)
    
    n_evaluation_samples = args.n_evaluations
    n_batch = args.batch_size
    kl_bound = args.kl_bound
    kl_context_bound = args.context_kl_bound
    if args.forward:
        kl_type = "forward"
    else:
        kl_type = "reverse"

    normalize = args.normalize

    print('+' * 100)
    print('parameters fitting !')
    
    if args.alg=='REPS':
        " imitation learning "
        imitation = CT_ImitationLearning(state_dim,
                                         parameters.shape[1] - set_params["alg_options"]["latent_dim"],
                                         set_params["alg_options"]["latent_dim"],
                                         n_clusters, use_dr=not args.not_dr)
        state_list = parameters[:, :state_dim]
        print("state_list :", state_list)
        context_list = parameters[:, state_dim:]
        print("params_list :", context_list)
        imitation.fit(parameters[:, :state_dim],
                      parameters[:, state_dim:],
                      forgetting_rate=args.forgetting_rate)
        
        rl_model = CT_ReinforcementLearning(imitation, kl_bound=kl_bound)
    elif args.alg=='MPPCA':
        " MPPCA "
        mppca = MPPCA(n_clusters, int(set_params["alg_options"]["latent_dim"]), n_init=5)
        mppca.fit(parameters)

        rl_model = RLModel(mppca,
                           context_dim=state_dim, kl_bound=kl_bound,
                           kl_bound_context=kl_context_bound, kl_reg=args.context_reg,
                           normalize=normalize,
                           kl_type=kl_type
                           )
        
        sr = Lampo(rl_model, wait=not args.slurm)
    else:
        print("Please give parameter fit model !")

    myplot = LampoMonitor(kl_bound, kl_context_bound=kl_context_bound,
                          title="class_log kl=%.2f, %d samples" %
                          (kl_bound, n_batch))
    
    print("rl_model :", rl_model)
    collector = RunModelMujoco(task, rl_model, args.dense_reward)
    
    reward_list = []
    for i in range(args.max_iter):
        s, r, actions, latent, cluster, observation = \
            collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)
         
        if args.alg == 'REPS':
            rl_model.add_dataset(observation[:n_batch], actions[:n_batch], r[:n_batch])
            print("ITERATION", i)
            print("SUCCESS:", np.mean(s))
    
        if args.alg == 'MPPCA':
            sr.add_dataset(actions[:n_batch], latent[:n_batch], cluster[:n_batch], observation[:n_batch], r[:n_batch])
            print("ITERATION", i)
            print("SUCCESS:", np.mean(s))
        
        reward_list.append(np.mean(s))
        
        myplot.notify_outer_loop(np.mean(s), np.mean(r))

        if args.alg == "MPPCA":
            sr.improve()
            # print("Optimization %f" % sr.rlmodel._f)
            # print("KL %f <= %f" % (sr.rlmodel._g, kl_bound))
            # if kl_context_bound> 0:
            #     print("KL context %f <= %f" % (sr.rlmodel._h, kl_context_bound))
        myplot.notify_inner_loop(0., 0., 0., 0.)

        if args.plot:
            myplot.visualize()

    s, r, actions, latent, cluster, observation = collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)
    
    print("ITERATION", args.max_iter)
    print("SUCCESS:", np.mean(s))
    myplot.notify_outer_loop(np.mean(s), np.mean(r))
    
    if args.plot:
        myplot.visualize(last=True)
    
    print("reward_list :", np.array(reward_list))
    np.save(args.alg + "_" + args.env_name + "_" + "reward_list.npy", np.array(reward_list))
    
    # if args.save:
    #     myplot.save(experiment_path + "result_%d.npz" % args.id)

    # import time
    # for i in range(100):
    #     print("+"* 50, i)
    #     task.reset()
    #     task._env.render()
    #     time.sleep(0.5)
    #     # while True:
    #     #     task._env.render()

    # context, target_pos, target_quat, target_euler = task.read_context()
    #
    # # set parameters :::
    # scale = np.array([2.0, 0.0, 0.0])
    # num_waypoints = 3
    # scale_list = scale.repeat([6, 6, 6], axis=0).reshape(num_waypoints, 6)
    # # print("scale_list :::", scale_list)
    #
    # stiffness_list = np.array([[10., 10., 6., 6., 6., 6.],
    #                            [0., 0., 0., 0., 0., 0.],
    #                            [0., 0., 0., 0., 0., 0.]])
    #
    # control_scale = np.ones_like(stiffness_list) * 10
    # stiffness_list = control_scale * stiffness_list
    # damping_list = scale_list * np.sqrt(stiffness_list)
    # weight_list = np.array([1.0, 0.0, 0.0])
    #
    # # pos_set_list = np.array([[-0.80349109, 0.09318907, 0.07348721],
    # #                          [-0.78349109, 0.09318907, 0.07348721],
    # #                          [-0.80349109, 0.09318907, -0.03348721]])
    #
    # # quat_set_list = np.array([[1.9265446606661554, -0.40240192959667226, -1.541555812071902],
    # #                           [1.9265446606661554, -0.40240192959667226, -1.541555812071902],
    # #                           [1.9265446606661554, -0.40240192959667226, -1.541555812071902]])
    #
    # pos_set_list = np.zeros((3, 3))
    # quat_set_list = np.zeros((3, 3))
    # for i in range(num_waypoints):
    #     pos_set_list[i, :] = target_pos
    #     quat_set_list[i, :] = target_euler
    #
    # way_points_list = np.concatenate((pos_set_list, quat_set_list), axis=1)
    # # print("way_point_list :::", way_points_list)
    #
    # task.set_waypoints(way_points_list)
    #
    # print("params :", stiffness_list.reshape(1, 18))
    # print("weight_list :", weight_list)
    # params = np.hstack((stiffness_list.reshape(1, 18)[0], weight_list))
    #
