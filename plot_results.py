import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
import numpy as np
import os
import re
import seaborn as sns

sns.set_theme()
# plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'Times New Roman'
X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


class Plotter:
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
              'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
              'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

    RETURN_TRAIN = 'episodic_return_train'
    RETURN_TEST = 'episodic_return_test'

    def __init__(self, root_path=None):
        self.root_path = root_path

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _window_func(self, x, y, window, func):
        yw = self._rolling_window(y, window)
        yw_func = func(yw, axis=-1)
        return x[window - 1:], yw_func
 
    def load_results(self, dirs, **kwargs):
        kwargs.setdefault('tag', self.RETURN_TRAIN)
        kwargs.setdefault('right_align', False)
        kwargs.setdefault('window', 0)
        kwargs.setdefault('top_k', 0)
        kwargs.setdefault('top_k_measure', None)
        kwargs.setdefault('interpolation', 100)
        xy_list = self.load_log_dirs(dirs, **kwargs)

        if kwargs['top_k']:
            perf = [kwargs['top_k_measure'](y) for _, y in xy_list]
            top_k_runs = np.argsort(perf)[-kwargs['top_k']:]
            new_xy_list = []
            for r, (x, y) in enumerate(xy_list):
                if r in top_k_runs:
                    new_xy_list.append((x, y))
            xy_list = new_xy_list

        if kwargs['interpolation']:
            x_right = float('inf')
            for x, y in xy_list:
                x_right = min(x_right, x[-1])
            x = np.arange(0, x_right, kwargs['interpolation'])
            y = []
            for x_, y_ in xy_list:
                y.append(np.interp(x, x_, y_))
            y = np.asarray(y)
        else:
            x = xy_list[0][0]
            y = [y for _, y in xy_list]
            x = np.asarray(x)
            y = np.asarray(y)

        return x, y

    def load_csv_results(self, dirs, **kwargs):
        kwargs.setdefault('tag', self.RETURN_TRAIN)
        kwargs.setdefault('right_align', False)
        kwargs.setdefault('window', 0)
        kwargs.setdefault('top_k', 0)
        kwargs.setdefault('top_k_measure', None)
        kwargs.setdefault('interpolation', 100)
        # xy_list = load_results(dirs)

        xy_list = []
        for folder in dirs:
            tslist = load_results(folder)
            xy_list.append(tslist)
        print("xy_list :", xy_list)

        if kwargs['top_k']:
            perf = [kwargs['top_k_measure'](y) for _, y in xy_list]
            top_k_runs = np.argsort(perf)[-kwargs['top_k']:]
            new_xy_list = []
            for r, (x, y) in enumerate(xy_list):
                if r in top_k_runs:
                    new_xy_list.append((x, y))
            xy_list = new_xy_list

        if kwargs['interpolation']:
            x_right = float('inf')
            for x, y in xy_list:
                x_right = min(x_right, x[-1])
            x = np.arange(0, x_right, kwargs['interpolation'])
            y = []
            for x_, y_ in xy_list:
                y.append(np.interp(x, x_, y_))
            y = np.asarray(y)
        else:
            x = xy_list[0][0]
            y = [y for _, y in xy_list]
            x = np.asarray(x)
            y = np.asarray(y)

        return x, y

    def filter_log_dirs(self, pattern, negative_pattern=' ', root='./log', **kwargs):
        dirs = [item[0] for item in os.walk(root)]
        leaf_dirs = []
        for i in range(len(dirs)):
            if i + 1 < len(dirs) and dirs[i + 1].startswith(dirs[i]):
                continue
            leaf_dirs.append(dirs[i])
        names = []
        p = re.compile(pattern)
        np = re.compile(negative_pattern)
        for dir in leaf_dirs:
            if p.match(dir) and not np.match(dir):
                names.append(dir)
                print(dir)
        print('')
        return sorted(names)

    def load_log_dirs(self, dirs, **kwargs):
        kwargs.setdefault('right_align', False)
        kwargs.setdefault('window', 0)
        xy_list = []
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for dir in dirs:
            event_acc = EventAccumulator(dir)
            event_acc.Reload()
            _, x, y = zip(*event_acc.Scalars(kwargs['tag']))
            xy_list.append([x, y])
        if kwargs['right_align']:
            x_max = float('inf')
            for x, y in xy_list:
                x_max = min(x_max, len(y))
            xy_list = [[x[:x_max], y[:x_max]] for x, y in xy_list]
        if kwargs['window']:
            xy_list = [self._window_func(np.asarray(x), np.asarray(y), kwargs['window'], np.mean) for x, y in xy_list]
        return xy_list

    def plot_mean(self, data, x=None, **kwargs):
        import matplotlib.pyplot as plt
        if x is None:
            x = np.arange(data.shape[1])
        if kwargs['error'] == 'se':
            e_x = np.std(data, axis=0) / np.sqrt(data.shape[0])
        elif kwargs['error'] == 'std':
            e_x = np.std(data, axis=0)
        else:
            raise NotImplementedError
        m_x = np.mean(data, axis=0)
        del kwargs['error']
        plt.plot(x, m_x, **kwargs)
        del kwargs['label']
        plt.fill_between(x, m_x + e_x, m_x - e_x, alpha=0.3, **kwargs)

    def plot_median_std(self, data, x=None, **kwargs):
        import matplotlib.pyplot as plt
        if x is None:
            x = np.arange(data.shape[1])
        e_x = np.std(data, axis=0)
        m_x = np.median(data, axis=0)
        plt.plot(x, m_x, **kwargs)
        del kwargs['label']
        plt.fill_between(x, m_x + e_x, m_x - e_x, alpha=0.3, **kwargs)

    def plot_games(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(l * 5, 5))
        for i, game in enumerate(games):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]

                # log_dirs = self.filter_log_dirs(pattern='.*%s.*%s.*%s' % (game, p, p), **kwargs)
                # print(log_dirs)

                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s.*%s' % (game, p, 'run'), **kwargs)
                print(log_dirs)

                # log_dirs = self.filter_log_dirs(pattern='.*%s.*%s.*%s.*%s' % (game, p, 'reps_mf', 'run'), **kwargs)
                # print(log_dirs)

                x, y = self.load_results(log_dirs, **kwargs)
                print("x_shape :::", x.shape)
                print("y_shape :::", y.shape)
                np.save(self.root_path + '/' + '%s_%s_reward.npy'%(game, p), y)

                # np.save('../plot_data/' + self.root_path + '/' + game + '/' + '%s_%s_reward.npy'%(game, p), y * kwargs['reward_scale'])

                # np.save('%s_reward.npy' % p, y)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xlabel('steps')
            if not i:
                plt.ylabel(kwargs['tag'])
            plt.title(game)
            plt.legend()

    def plot_specifications(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(l * 5, 5))
        for i, game in enumerate(games):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s.*%s' % (game, p, 'run'), **kwargs)
                # log_dirs = self.filter_log_dirs(pattern='.*%s.*%s' % (game, p), **kwargs)
                # log_dirs = self.root_path + '/' + game + '/' + p
                print("log_dirs :", log_dirs)
                # x, y = self.load_csv_results(log_dirs, **kwargs)
                x, y = self.load_results(log_dirs, **kwargs)
                # y = y * kwargs['reward_scale'][game]
                y = y * kwargs['reward_scale']

                if not os.path.exists('../plot_data/' + self.root_path + '/' + game):
                    os.makedirs('../plot_data/' + self.root_path + '/' + game)

                np.save('../plot_data/' + self.root_path + '/' + game + '/' + '%s_%s_'% (game, p) + kwargs['labels_y_axis'] + '.npy', y)

                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xlabel('steps')
            if not i:
                plt.ylabel(kwargs['tag'])

            plt.title(game)
            plt.legend()

    def plot_different(self, games, **kwargs):
        kwargs.setdefault('agg', 'mean')
        import matplotlib.pyplot as plt
        l = len(games)
        plt.figure(figsize=(l * 5, 5))
        for i, game in enumerate(games):
            plt.subplot(1, l, i + 1)
            for j, p in enumerate(kwargs['patterns']):
                label = kwargs['labels'][j]
                color = self.COLORS[j]
                log_dirs = self.filter_log_dirs(pattern='.*%s.*%s.*%s' % (game, p, p), **kwargs)
                print(log_dirs)
                x, y = self.load_results(log_dirs, **kwargs)
                print("x_shape :::", x.shape)
                print("y_shape :::", y.shape)
                np.save('%s%s_reward.npy' % (game, p), y)
                if kwargs['downsample']:
                    indices = np.linspace(0, len(x) - 1, kwargs['downsample']).astype(np.int)
                    x = x[indices]
                    y = y[:, indices]
                if kwargs['agg'] == 'mean':
                    self.plot_mean(y, x, label=label, color=color, error='se')
                elif kwargs['agg'] == 'mean_std':
                    self.plot_mean(y, x, label=label, color=color, error='std')
                elif kwargs['agg'] == 'median':
                    self.plot_median_std(y, x, label=label, color=color)
                else:
                    for k in range(y.shape[0]):
                        plt.plot(x, y[i], label=label, color=color)
                        label = None
            plt.xlabel('steps')
            if not i:
                plt.ylabel(kwargs['tag'])
            plt.title(game)
            plt.legend()

    def select_best_parameters(self, patterns, **kwargs):
        scores = []
        for pattern in patterns:
            log_dirs = self.filter_log_dirs(pattern, **kwargs)
            xy_list = self.load_log_dirs(log_dirs, **kwargs)
            y = np.asarray([xy[1] for xy in xy_list])
            scores.append(kwargs['score'](y))
        indices = np.argsort(-np.asarray(scores))
        return indices


def plot_reward_results(root_path,
                        task_list=['insertion', 'push', 'hammer']
                        ):
    plotter = Plotter(root_path=root_path)

    games = [
        'push',
        'insertion',
        'hammer'
    ]

    patterns = [
        # 'PPO2',
        # 'SAC',
        # 'TD3'
        'HPSSAC'
    ]

    labels = [
        # 'PPO2',
        # 'SAC',
        # 'TD3'
        'HPSSAC'
    ]

    tags = [
        # 'success'
        "episode_reward"
    ]

    reward_scale = {
        "insertion": 1,
        "push": 0.1,
        "hammer": 0.01
    }

    # reward_scale = {
    #     "insertion": 1,
    #     "push": 1,
    #     "hammer": 1
    # }
    for i, task in enumerate(task_list):
        plotter.plot_games(games=[task],
                           patterns=patterns,
                           agg='mean_std',
                           downsample=1000,
                           labels=labels,
                           right_align=False,
                           tag=tags[0],
                           root=root_path,
                           interpolation=100,
                           window=100,
                           reward_scale=reward_scale[games[i]]
                           )

    plt.tight_layout()
    # plt.savefig(root_path + 'env_SAC_PPO2_reward.png', bbox_inches='tight')
    plt.show()


def plot_specific_results(root_path,
                          task_list=['insertion', 'push', 'hammer']
                          ):
    plotter = Plotter(root_path=root_path)
    
    games = [
        'insertion',
        'push',
        'hammer'
    ]
    
    patterns = [
        # 'PPO2'
        # 'SAC',
        # 'TD3',
        # 'HPSSAC'
        'HPSTD3'
    ]

    labels = [
        # 'PPO2'
        # 'SAC',
        # 'TD3',
        # 'HPSSAC'
        'HPSTD3'
    ]

    tags = {
        "insertion": "eval/success/final",
        "push": "eval/block_pos_dist/final",
        "hammer": "eval/nail_depth/final"
    }

    reward_scale = {
        "insertion": 1,
        "push": 1,
        "hammer": 1
    }

    labels_y_axis = {
        "insertion": "success",
        "push": "block_pos_dist",
        "hammer": "nail_depth"
    }
    
    for i, task in enumerate(task_list):
        index = i
        plotter.plot_specifications(games=[task],
                                    patterns=patterns,
                                    agg='mean',
                                    downsample=100,
                                    labels=labels,
                                    right_align=False,
                                    # tag=tags[game],
                                    tag=tags[games[index]],
                                    root=root_path,
                                    interpolation=100,
                                    window=100,
                                    reward_scale=reward_scale[games[index]],
                                    labels_y_axis=labels_y_axis[games[index]]
                                    )

    # plotter.plot_specifications(games=games,
    #                             patterns=patterns,
    #                             agg='mean',
    #                             downsample=100,
    #                             labels=labels,
    #                             right_align=False,
    #                             tag=tags["insertion"],
    #                             root=root_path,
    #                             interpolation=100,
    #                             window=100,
    #                             )

    plt.tight_layout()
    # plt.savefig(root_path + 'env_SAC_PPO2_reward.png', bbox_inches='tight')
    plt.show()


def plot_multi_reward_results(dirs,
                              label_list,
                              fig_path=None,
                              title=None
                              ):
    plt.figure(figsize=(5, 4))

    for i in range(len(dirs)):
        data = np.load(dirs[i])
        length = data.shape[1]
        print(data.shape)
        # if i == 0:
        #     plt.plot(np.arange(length), np.mean(data, axis=0)+150, label=label_list[i])
        #     plt.fill_between(np.arange(length),
        #                      np.mean(data, axis=0)+150 - np.std(data, axis=0),
        #                      np.mean(data, axis=0)+150 + np.std(data, axis=0), alpha=0.5)
        # else:
        #     plt.plot(np.arange(length), np.mean(data, axis=0), label=label_list[i])
        #     plt.fill_between(np.arange(length),
        #                      np.mean(data, axis=0) - np.std(data, axis=0),
        #                      np.mean(data, axis=0) + np.std(data, axis=0), alpha=0.5)

        plt.plot(np.arange(length), np.mean(data, axis=0), label=label_list[i])
        plt.fill_between(np.arange(length),
                         np.mean(data, axis=0) - np.std(data, axis=0),
                         np.mean(data, axis=0) + np.std(data, axis=0), alpha=0.5)

    plt.plot([0, 5000], [200, 200], linestyle='--')

    # plt.xlim(minx, maxx)
    plt.title(title)
    # plt.xlabel(r"Time Steps ($\times$ 100)")
    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000], (0, 500, 1000, 1500, 2000, 2500))
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend()
    plt.tight_layout()
    # plt.savefig(fig_path + 'PPO_SAC.png')
    plt.savefig(fig_path + title + '_PPO_SAC.pdf', dpi=200)
    plt.show()


def plot_tii_multi_reward_results(dirs,
                                  label_list,
                                  title,
                                  fig_path=''):
    plt.figure(figsize=(4, 3))
    for i in range(len(dirs)):
        data = np.load(dirs[i])
        length = data.shape[1]
        print("data :", data)
        print(data.shape)
        # if i == 0:
        #     plt.plot(np.arange(length), np.mean(data, axis=0), label=label_list[i])
        #     plt.fill_between(np.arange(length),
        #                      np.mean(data, axis=0) - np.std(data, axis=0),
        #                      np.mean(data, axis=0) + np.std(data, axis=0), alpha=0.5)
        # else:
        plt.plot(np.arange(length), np.mean(data, axis=0), label=label_list[i])
        plt.fill_between(np.arange(length),
                         np.mean(data, axis=0) - np.std(data, axis=0),
                         np.mean(data, axis=0) + np.std(data, axis=0), alpha=0.8)

    # plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(r"Time Steps ($\times$ 100)")
    # plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    # plt.xticks([0, 1000, 2000, 3000, 4000, 5000], (0, 500, 1000, 1500, 2000, 2500))
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + 'PPO_SAC.png', bbox_inches='tight')
    plt.show()


def plot_multi_tasks_reward(games=['insertion'],
                            methods=['SAC'],
                            root_path_list=['com_vices'],
                            label_list=None):
    l = len(games)
    sns.set(font_scale=1.3)

    plt.figure(figsize=(l * 5, 4))

    for i, game in enumerate(games):
        plt.subplot(1, l, i + 1)
        # for j, method in enumerate(methods):
        #     path = root_path + '/' + game + '/' + game + '_' + method + '_reward.npy'
        #     reward = np.load(path)
        #     length = reward.shape[1]
        #     plt.plot(np.arange(length), np.mean(reward, axis=0), label=label_list[j])
        #     plt.fill_between(np.arange(length),
        #                      np.mean(reward, axis=0) - np.std(reward, axis=0),
        #                      np.mean(reward, axis=0) + np.std(reward, axis=0), alpha=0.5)

        for j, root_path in enumerate(root_path_list):
            path = root_path + '/' + game + '/' + game + '_' + 'SAC' + '_reward.npy'
            # path = root_path + '/' + game + '/' + game + '_' + 'SAC' + '_train_episode_reward.npy'
            reward = np.load(path)
            length = reward.shape[1]
            plt.plot(np.arange(length), np.mean(reward, axis=0), label=label_list[j])
            plt.fill_between(np.arange(length),
                             np.mean(reward, axis=0) - np.std(reward, axis=0),
                             np.mean(reward, axis=0) + np.std(reward, axis=0), alpha=0.5)


        plt.title(game)
        plt.xlabel(r"Time Steps ($\times$ 100)")
        plt.ylabel("Episode Reward")

        plt.legend()

    plt.tight_layout()
    # plt.savefig('PPO_SAC.png')
    plt.show()


def plot_multi_tasks_info(games=['insertion'],
                          methods=['SAC'],
                          root_path_list=['com_vices'],
                          label_list=None):
    l = len(games)
    sns.set(font_scale=1.3)
    info_results = {
                    "insertion": "success",
                    "push": "block_pos_dist",
                    "hammer": "nail_depth"
                    }

    plt.figure(figsize=(l * 5, 4))

    for i, game in enumerate(games):
        plt.subplot(1, l, i + 1)
        # for j, method in enumerate(methods):
        #     path = root_path + '/' + game + '/' + game + '_' + method + '_reward.npy'
        #     reward = np.load(path)
        #     length = reward.shape[1]
        #     plt.plot(np.arange(length), np.mean(reward, axis=0), label=label_list[j])
        #     plt.fill_between(np.arange(length),
        #                      np.mean(reward, axis=0) - np.std(reward, axis=0),
        #                      np.mean(reward, axis=0) + np.std(reward, axis=0), alpha=0.5)

        for j, root_path in enumerate(root_path_list):
            path = root_path + '/' + game + '/' + game + '_' + 'SAC' + '_' + info_results[game] + '.npy'
            # path = root_path + '/' + game + '/' + game + '_' + 'SAC' + '_train_episode_reward.npy'
            reward = np.load(path)
            length = reward.shape[1]
            plt.plot(np.arange(length), np.mean(reward, axis=0), label=label_list[j])
            plt.fill_between(np.arange(length),
                             np.mean(reward, axis=0) - np.std(reward, axis=0),
                             np.mean(reward, axis=0) + np.std(reward, axis=0), alpha=0.5)


        plt.title(game)
        plt.xlabel(r"Time Steps ($\times$ 100)")
        # plt.ylabel("Episode Reward")
        plt.ylabel(info_results[game])

        plt.legend()

    plt.tight_layout()
    # plt.savefig('PPO_SAC.png')
    plt.show()


def plot_assitive_reward_results(root_path):
    plotter = Plotter(root_path=root_path)
    games = [
        # 'DrinkingSawyerHuman-v1/fixed',
        'FeedingSawyerHuman-v1/fixed',
        # 'ScratchItchJacoHuman-v1/fixed'
        # 'insertion'
        # 'hammer'
    ]

    patterns = [
        'PPO',
        # 'SAC'
    ]

    labels = [
        'PPO',
        # 'SAC'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       # tag='episode_reward',
                       tag='train_episode_reward',
                       root=root_path,
                       interpolation=100,
                       window=10,
                       reward_scale=1,
                       )

    plt.tight_layout()
    # plt.savefig(root_path + 'env_assitive_SAC_PPO2_reward.png', bbox_inches='tight')
    plt.show()


def rolling_window(array, window):
    """
        apply a rolling window to a np.ndarray

        :param array: (np.ndarray) the input Array
        :param window: (int) length of the rolling window
        :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


def ts2xy(timesteps, xaxis):
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        x_var = np.arange(len(timesteps))
        y_var = timesteps[xaxis].values

    return x_var, y_var


def plot_curves(xy_list, xaxis, yaxis, title):
    """
        plot the curves

        xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
        xaxis: (str) the axis for the x and y output
            (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
        title: (str) the title of the plot
    """
    plt.figure(figsize=(5, 4))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    x_list = []
    y_list = []
    min_x_label = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        # plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            x_list.append(x)
            y_list.append(y_mean)

            if len(x_list[min_x_label]) < len(x):
                min_x_label = i
            else:
                pass

            # plt.plot(x, y_mean, color=color)

    print("y_list :", y_list)
    print("y_list_shape :", np.array(y_list))
    print(len(x_list[min_x_label]))
    real_data_y = np.array(y_list)[:, :len(x_list[min_x_label])]
    print("real_data_shape :", real_data_y)
    plt.plot(x_list[min_x_label], np.mean(real_data_y, axis=0), color=COLORS[0])
    plt.fill_between(x_list[min_x_label],
                     np.mean(real_data_y, axis=0) - np.std(real_data_y, axis=0),
                     np.mean(real_data_y, axis=0) + np.std(real_data_y, axis=0), alpha=0.5)

    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()


def plot_info_results(result_path_list=['./com_vices_tii/'],
                      policy_name='SAC',
                      env_name='push'
                      ):
    info_key_list = {
        "hammer": "nail_depth",
        "insertion": "success",
        "push": "block_pos_dist"
    }
    info_key = info_key_list[env_name]
    folder_num = 5
    num_timesteps = 1e6
    dir_list = []
    for k in range(len(result_path_list)):
        result_path = result_path_list[k]
        dirs = []
        for i in range(folder_num):
            dirs.append(result_path + env_name + '/' + policy_name + '/' + 'run_%s' % str(i))
            # dirs = ['./comparison_tmech/insertion/PPO2/run_0', './comparison_tmech/insertion/PPO2/run_1',
            #         './comparison_tmech/insertion/PPO2/run_2', './comparison_tmech/insertion/PPO2/run_3']
        dir_list.append(dirs)
        print("dirs", dirs)

    # plot_results(dirs, args.episode_reward, info_key, args.task_name)
    for k in range(len(dir_list)):
        dirs = dir_list[k]
        tslist = []
        for folder in dirs:
            timesteps = load_results(folder)
            if num_timesteps is not None:
                timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
            tslist.append(timesteps)

    xy_list = [ts2xy(timesteps_item, info_key) for timesteps_item in tslist]

    plot_curves(xy_list, 'Episodes', info_key, env_name)

    plt.tight_layout()
    plt.show()


def plot_ori_assistive_reward(
        algo_list=None,
        env_name=None,
        label_list=None
):
    sns.set(font_scale=1.3)
    plt.figure(figsize=(5, 4))
    index = int(1912/53)
    for i in range(len(algo_list)):
        algo = algo_list[i]
        reward = np.load('./trained_models_old/' + algo + '/' + env_name + '/training_reward.npy')
        print("reward :", reward)
        reward_min = np.load('./trained_models_old/' + algo + '/' + env_name + '/training_reward_min.npy')
        reward_max = np.load('./trained_models_old/' + algo + '/' + env_name + '/training_reward_max.npy')
        if algo == 'sac':
            print("reward :", reward.shape)
            reward = reward[::index][:53]
            reward_min = reward_min[::index][:53]
            reward_max = reward_max[::index][:53]
        plt.plot(reward, label=label_list[i])
        plt.fill_between(np.arange(reward.shape[0]), reward_min, reward_max, alpha=0.5)
    
    plt.xlabel(r"Time Steps ($1\times10^3$)")
    plt.ylabel("Episode Reward")
    plt.xticks([0, 10, 20, 30, 40, 50], [0, 200, 400, 600, 800, 1000])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Pure_RL_' + env_name + '_PPO_SAC.pdf')
    plt.show()
    
    
if __name__ == '__main__':
    # ================================================================================
    # =================== robotic healthcare assistive tasks =========================
    # ================================================================================
    # task = ['FeedingSawyerHuman-v1', 'DrinkingSawyerHuman-v1', 'ScratchItchJacoHuman-v1']
    # title = ['Feeding', 'Drinking']
    # controller = 'fixed'
    # algorithm_list = ['SAC', 'PPO2']
    #
    # # plot_assitive_reward_results(root_path='results/5-assistive/ImpedanceDRL')
    #
    # for index in range(0, 1):
    #     # index = 2
    #     plot_multi_reward_results([
    #         'results/5-assistive/ImpedanceDRL/' + task[index] + '/' + 'fixed_' + algorithm_list[0] + '_reward.npy',
    #         'results/5-assistive/ImpedanceDRL/' + task[index] + '/' + 'fixed_' + algorithm_list[1] + '_reward.npy'],
    #         label_list=algorithm_list,
    #         fig_path='results/5-assistive/ImpedanceDRL/' + task[index] + '/',
    #         title=title[index]
    #     )
    
    # plot_multi_reward_results([
    #     './assitive/' + task + '/' + controller + '/' + algorithm + '_reward.npy',
    #     './com_vices/' + task + '/' + task + '_' + algorithm + '_reward.npy',
    #     './com_context_iros/' + task + '/' + task + '_' + algorithm + '_reward.npy'],
    #     label_list=['IROS', 'VICES', 'Fixed']
    # )

    # ================================================================================
    # folder_list = [
    #     './com_vices_variable_context_tii',
    #     './com_vices_fixed_context_tii',
    #     './com_vices_pure_tii',
    #     './com_cps_tii',
    #     './com_vices',
    #     './com_vic',
    #     './com_iros'
    # ]
    #
    # task_list = ['insertion-low-ime',
    #              'insertion-normal-ime',
    #              'insertion-large-ime']
    #
    # # task_list = ['hammer', 'insertion', 'push']
    # algorithm = ['SAC', 'PPO2', 'TD3']
    # label_list = ['VIC', 'VICES', 'REPS', 'TD3']
    # index = 2

    # plot_tii_multi_reward_results([
    #     # './' + folder_list[1] + '/' + task_list[index] + '/' + task_list[index] + '_' + algorithm[0] + '_reward.npy',
    #     './' + folder_list[3] + '/' + task_list[index] + '/' + task_list[index] + '_' + algorithm[1] + '_reward.npy'
    #     # './com_vices/' + task + '/' + task + '_' + algorithm + '_reward.npy',
    #     # './com_context_iros/' + task + '/' + task + '_' + algorithm + '_reward.npy'
    #     ],
    #     label_list=algorithm[:1],
    #     title=task_list[index]
    # )

    # plot_multi_tasks_reward(games=task_list,
    #                         methods=algorithm[2],
    #                         root_path_list=folder_list[2],
    #                         label_list=label_list[3]
    #                         )

    # plot_multi_tasks_info(games=task_list,
    #                       methods=algorithm[0],
    #                       root_path_list=folder_list[:3],
    #                       label_list=label_list[:3]
    #                       )
    # ================================================================================

    # reward_hammer = np.load('com_context_iros/insertion/insertion_PPO2_reward.npy')
    # # # np.save('com_iros/hammer/hammer_SAC_reward.npy', reward_hammer * 0.1)
    # print("reward_hammer :", reward_hammer.shape)

    # =================================================================
    # plot_reward_results(root_path='com_vic_stable_variable')
    # plot_reward_results(root_path='com_vices_stable_variable')
    # plot_reward_results(root_path='./com_context_iros/')
    # plot_reward_results(root_path='./com_vices_pure_tii/')

    # =================================================================
    # plot_specific_results(root_path='./com_vices_pure_tii')
    # plot_specific_results(root_path='./com_vices_fixed_context_tii')
    # plot_specific_results(root_path='com_vic_stable_variable',
    #                       task_list=['insertion-low-ime',
    #                                  'push-low-ime',
    #                                  'hammer-low-ime'
    #                       ]
    #                       )
    
    # plot_reward_results(root_path='com_hps_stable_variable',
    #                     task_list=[
    #                                  'insertion-low-ime-range'
    #                     ])
    
    # plot_specific_results(root_path='com_hps_stable_variable',
    #                       task_list=[
    #                                  'insertion-low-ime-range'
    #                       ]
    #                       )
    
    # plot_specific_results(root_path='com_vic_stable_variable',
    #                       task_list=['insertion-low-ime',
    #                                  'push-low-ime',
    #                                  'hammer-low-ime'])
    
    # =================================================================
    # import pickle
    # with open_path(path, "r", suffix="pkl") as file_handler:
    # df = open('run_5replay_buffer.pkl', 'rb')
    # data = pickle.load(df)
    # print("data :::", data)

    # plot_info_results()

    # plot_specific_results(root_path='./com_vices_tii/')

    # plotter_1.plot_envs(dirs=['./comparison_tmech/hammer/SAC/SAC_1/', './comparison_tmech/hammer/SAC/SAC_2/'],
    #                     tag='episode_reward',
    #                     agg='median',
    #                     patterns='SAC',
    #                     labels=['A', 'B'],
    #                     downsample=0,
    #                     right_align=False,
    #                     )
    algo_list = ['ppo', 'sac']
    env_name_list = ['FeedingSawyer-v1', 'DrinkingSawyer-v1']

    plot_ori_assistive_reward(
        algo_list=algo_list,
        env_name=env_name_list[1],
        label_list=['PPO', 'SAC']
    )