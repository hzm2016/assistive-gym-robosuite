import argparse


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
                        default=30)
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
                        default=50)
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
    parser.add_argument('--num_restarts',
                        type=int,
                        default=1,
                        help='The number of trials to run.')
    
    parser.add_argument("--env",
                        default="FeedingSawyerHuman-v1")  # OpenAI gym environment name
    
    parser.add_argument("-r",
                        "--slurm",
                        help="Don't look for CPU usage.",
                        action="store_true")
    parser.add_argument("--forward",
                        help="Bound the improvement kl.",
                        default=False)
    
    parser.add_argument("--alg",
                        default='MPPCA')  # REPS
    
    parser.add_argument("--load_data",
                        default=False)
    
    args = parser.parse_args()
    return args