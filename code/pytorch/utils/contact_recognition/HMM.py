import pickle
import time
import numpy as np
from hmmlearn import hmm

# x_dim = 4
# y_dim = 3
# X = np.array([[10, 20, 30, 40, 1.1, 1.2, 1.3],
#               [20, 30, 40, 50, 1.2, 1.3, 1.4],
#               [30, 40, 50, 60, 1.3, 1.4, 1.5],
#               [40, 50, 60, 70, 1.4, 1.5, 1.6],
#               [50, 60, 70, 80, 1.5, 1.6, 1.7],
#               [60, 70, 80, 90, 1.6, 1.7, 1.8],
#               [70, 80, 90, 100, 1.7, 1.8, 1.9],
#               [80, 90, 100, 110, 1.8, 1.9, 2.0],
#               [90, 100, 110, 120, 1.9, 2.0, 2.1],
#               [100, 110, 120, 130, 2.0, 2.1, 1.3],
#               [110, 120, 130, 140, 2.1, 2.2, 1.4],
#               [120, 130, 140, 150, 2.2, 2.3, 1.5],
#               [130, 140, 150, 160, 2.3, 2.4, 1.6],
#               [140, 150, 160, 170, 2.4, 2.5, 1.7],
#               [150, 160, 170, 180, 2.5, 2.6, 1.8],
#               [160, 170, 180, 190, 2.6, 2.7, 2.8],
#               [170, 180, 190, 200, 2.7, 2.8, 2.9],
#               [180, 190, 200, 210, 2.8, 2.9, 3.0]])
# X = np.vstack([X, X])
# X_noise = np.hstack([np.random.randn(36, x_dim)*10, np.random.randn(36, y_dim)/10])
# X = X + X_noise
#
# X_train = np.array([X.ravel()]).transpose()
# Y_train = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]
#
#
# clf = hmm.GaussianHMM(n_components=3, n_iter=1000, tol=0.01,covariance_type="full").fit(X_train, Y_train)
#
# x = np.array([[120, 130, 140, 150, 2.2, 2.3, 1.5]]).transpose()
# p = clf.score(x)
# print(p)


class Gaussian_HMM_Classification():
    def __init__(self, phase=5, components=3):
        self.memory = []
        self.buffer = []
        self.buffer_ = []
        self.f_n = phase
        self.c_n = components
        self.HMM = []
        print('Gaussian_HMM initialized, phase number:', phase, ', GM component number:', components)

    def train(self, memory_train):
        self.memory = memory_train
        print('total training data size:', len(self.memory))

        # distribute the data in memory into each phase's smaller buffer
        self.buffer = []
        self.buffer_ = []
        for i in range(self.f_n):
            self.buffer.append([])

        for i in range(len(self.memory)):
            obs = self.memory[i][1:]
            self.buffer[int(self.memory[i][0] - 1)].append(obs)

        for i in range(len(self.buffer)):
            print('data size of phase', i+1, ':', len(self.buffer[i]))
            y = [len(self.buffer[i][0])] * len(self.buffer[i])
            self.buffer[i] = np.array([np.array(self.buffer[i]).ravel()]).transpose()
            self.buffer_.append(y)

        # train
        t = time.time()
        T = time.time()
        self.HMM = []
        for i in range(self.f_n):
            self.HMM.append(hmm.GaussianHMM(n_components=3, n_iter=1000, tol=0.01, covariance_type="full").fit(self.buffer[i], self.buffer_[i]))
            print('HMM of phase', i + 1, 'trained, training time:', time.time() - t)
            t = time.time()
        print('total training time:', time.time() - T)

        # record the model
        f = open('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master//HMM_model.pckl', 'wb')
        pickle.dump(self.HMM, f)
        f.close()

    def phase_recognition(self, x):  # input one_dimension array
        x = np.array([x]).transpose()
        P = []
        for i in range(self.f_n):
            P.append(self.HMM[i].score(x))
        return P.index(max(P)) + 1

    def classification_comparation(self, memory_test):
        samples = memory_test
        t = 0
        T = []
        for i in range(len(samples)):
            correct_phase = int(samples[i][0])
            computed_phase = self.phase_recognition(samples[i][1:])
            if correct_phase == computed_phase:
                t += 1
                T.append(1)
            else:
                T.append(0)
        print('success rate:', t/len(samples))
        return t/len(samples)
        # plt.plot(T)
        # plt.show()


# a = Gaussian_HMM_Classification()
# a.train()
# a.classification_comparation('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master//replay_buffer_small_test.pckl')