import numpy as np
import math
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib.pyplot as plt

# cov = np.eye(2, 2)
# u1 = np.array([2, 2])
# u2 = np.array([-2, -2])
# X1 = np.random.multivariate_normal(mean=u1, cov=cov, size=500)
# X2 = np.random.multivariate_normal(mean=u2, cov=cov, size=500)
# X = np.vstack([X1, X2])


class Gaussian_Mixture_Classification():
    def __init__(self, phase=5, components=3):
        self.memory = []
        self.buffer = []
        self.f_n = phase
        self.c_n = components
        self.GMM = []
        print('Gaussian_Mixture initialized, phase number:', phase, ', GM component number:', components)

    def train(self, memory_train):  # input a two_dimensional array
        self.memory = memory_train
        print('total training data size:', len(self.memory))

        # distribute the data in memory into each phase's smaller buffer
        self.buffer = []
        for i in range(self.f_n):
            self.buffer.append([])

        for i in range(len(self.memory)):
            obs = self.memory[i][1:]
            self.buffer[int(self.memory[i][0] - 1)].append(obs)
        print('data size of phase 1:', len(self.buffer[0]))
        print('data size of phase 2:', len(self.buffer[1]))
        print('data size of phase 3:', len(self.buffer[2]))
        print('data size of phase 4:', len(self.buffer[3]))
        print('data size of phase 5:', len(self.buffer[4]))

        # train
        t = time.time()
        T = time.time()
        self.GMM = []
        for i in range(self.f_n):
            self.GMM.append(mixture.GaussianMixture(n_components=self.c_n, covariance_type='full').fit(self.buffer[i]))
            print('GMM of phase', i + 1, 'trained, training time:', time.time() - t)
            t = time.time()
        print('total training time:', time.time() - T)

    def phase_recognition(self, x):  # input a one_dimensional array
        x = np.array([x])
        P = []
        for i in range(self.f_n):
            # u = self.GMM[i].means_
            # print(u)
            # cov = self.GMM[i].covariances_
            # weights = self.GMM[i].weights_
            # likelihood = 0.
            # for j in range(self.c_n):
            #     prob = 1/(((2*np.pi)**7.5)*(np.linalg.det(cov[j]))**0.5)*np.exp(-0.5*np.dot(np.dot((x-np.array([u[j]])), np.linalg.inv(cov[j])), (x-np.array([u[j]])).transpose()))
            #     likelihood += weights[j]*prob
            #     print(prob)
            # log_like = np.log(likelihood)
            # P.append(log_like)
            P.append(self.GMM[i].score(x))
        return P.index(max(P))+1

    def classification_comparation(self, memory_test):  # input a two_dimensional array
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


# x = np.array([-0.002,  0.0001,  0.058, -0.028,  0.0008,
#               -0.042, 0., 0., 0., -0.00453, 0., 0.00127, -0.00064, -0.008, 0.0008])

# h = Gaussian_Mixture_Classification()
# h.train()
# h.classification_comparation('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master//replay_buffer_small_test.pckl')







