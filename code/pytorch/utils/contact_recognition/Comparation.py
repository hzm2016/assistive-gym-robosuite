import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from Classification.GMM.GMC import Gaussian_Mixture_Classification as GMM
from Classification.HMM import Gaussian_HMM_Classification as HMM


h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes",
         "QDA", "Gaussian Process"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GaussianProcessClassifier(1.0 * RBF(1.0))]

path_train = 'replay_buffer_train.pckl'
f = open(path_train, 'rb')
memory_train = pickle.load(f)
memory_train = np.array(memory_train)
# add a small noise
pos_noise1 = np.random.randn(len(memory_train), 3)*0.001/2
rot_noise = np.random.randn(len(memory_train), 3)*0.003/2
lin_vel_noise = np.random.randn(len(memory_train), 3)*0.0004/2
rot_vel_noise = np.random.randn(len(memory_train), 3)*0.0012/2
pos_noise2 = np.random.randn(len(memory_train), 3)*0.001/2
noise = np.concatenate((np.zeros([len(memory_train), 1]), pos_noise1, rot_noise, lin_vel_noise, rot_vel_noise, pos_noise2), axis=1)
memory_train = memory_train + noise

X_train = memory_train[:, 1:]
y_train = memory_train[:, 0: 1].transpose()[0]


path_test = 'replay_buffer_test.pckl'
f = open(path_test, 'rb')
memory_test = pickle.load(f)
memory_test = np.array(memory_test)
# add a big noise
pos_noise1 = np.random.randn(len(memory_test), 3)*0.001*2
rot_noise = np.random.randn(len(memory_test), 3)*0.003*2
lin_vel_noise = np.random.randn(len(memory_test), 3)*0.0004*2
rot_vel_noise = np.random.randn(len(memory_test), 3)*0.0012*2
pos_noise2 = np.random.randn(len(memory_test), 3)*0.001*2
noise = np.concatenate((np.zeros([len(memory_test), 1]), pos_noise1, rot_noise, lin_vel_noise, rot_vel_noise, pos_noise2), axis=1)
memory_test = memory_test + noise

X_test = memory_test[:, 1:]
y_test = memory_test[:, 0: 1].transpose()[0]



# iterate over classifiers
# GMM
gmm = GMM(5, 3)
t0 = time.time()
gmm.train(memory_train)
gmm_t = time.time() - t0
gmm_score = gmm.classification_comparation(memory_test)

# HMM
hmm = HMM(5, 3)
t0 = time.time()
hmm.train(memory_train)
hmm_t = time.time() - t0
hmm_score = hmm.classification_comparation(memory_test)

print('')
print('***********************  Classification Results *************************')
print('GMM', ' accuracy:', gmm_score, ' training time:', gmm_t)
print('HMM', ' accuracy:', hmm_score, ' training time:', hmm_t)

# Other methods
for name, clf in zip(names, classifiers):
    t0 = time.time()
    clf.fit(X_train, y_train)
    t = time.time() - t0
    score = clf.score(X_test, y_test)
    print(name, ' accuracy:', score, ' training time:', t)
