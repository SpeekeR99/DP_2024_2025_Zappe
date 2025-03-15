import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.datasets import load_iris
# https://github.com/ngoix/EMMV_benchmarks.git
from lib.eval.em import em, mv

# This example is only an updated version of old example from https://github.com/ngoix/EMMV_benchmarks.git

# Parameters for the algorithm
n_generated = 100000
alpha_min = 0.9
alpha_max = 0.999
t_max = 0.9
ocsvm_max_train = 10000

np.random.seed(42)

if not os.path.exists(f"img/eval"):
    os.makedirs(f"img/eval")

# Load the data
data = load_iris()
X, y = data.data, data.target

n_samples, n_features = np.shape(X)
n_samples_train = n_samples // 2
n_samples_test = n_samples - n_samples_train

X_train = X[:n_samples_train, :]
X_test = X[n_samples_train:, :]
y_train = y[:n_samples_train]
y_test = y[n_samples_train:]

# Define models
iforest = IsolationForest()
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
ocsvm = OneClassSVM()

lim_inf = X.min(axis=0)
lim_sup = X.max(axis=0)
volume_support = (lim_sup - lim_inf).prod()
t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features))

# Fit the models
iforest.fit(X_train)
lof.fit(X_train)
ocsvm.fit(X_train[:min(ocsvm_max_train, n_samples_train - 1)])

s_X_iforest = iforest.decision_function(X_test)
s_X_lof = lof.decision_function(X_test)
s_X_ocsvm = ocsvm.decision_function(X_test).reshape(1, -1)[0]

s_unif_iforest = iforest.decision_function(unif)
s_unif_lof = lof.decision_function(unif)
s_unif_ocsvm = ocsvm.decision_function(unif).reshape(1, -1)[0]

# Get the metrics
em_auc_iforest, em_iforest, amax_iforest = em(t, t_max, volume_support, s_unif_iforest, s_X_iforest, n_generated)
em_auc_lof, em_lof, amax_lof = em(t, t_max, volume_support, s_unif_lof, s_X_lof, n_generated)
em_auc_ocsvm, em_ocsvm, amax_ocsvm = em(t, t_max, volume_support, s_unif_ocsvm, s_X_ocsvm, n_generated)

if amax_iforest == -1 or amax_lof == -1 or amax_ocsvm == -1:
    amax = -1
else:
    amax = max(amax_iforest, amax_lof, amax_ocsvm)

mv_auc_iforest, mv_iforest = mv(axis_alpha, volume_support, s_unif_iforest, s_X_iforest, n_generated)
mv_auc_lof, mv_lof = mv(axis_alpha, volume_support, s_unif_lof, s_X_lof, n_generated)
mv_auc_ocsvm, mv_ocsvm = mv(axis_alpha, volume_support, s_unif_ocsvm, s_X_ocsvm, n_generated)

# Plot the results
fig = plt.figure(figsize=(15, 10))

plt.subplot(121)
plt.plot(t[:amax], em_iforest[:amax], lw=1, label=f'Isolation Forest (EM-score = {em_auc_iforest:.3e})')
plt.plot(t[:amax], em_lof[:amax], lw=1, label=f'Local Outlier Factor (EM-score = {em_auc_lof:.3e})')
plt.plot(t[:amax], em_ocsvm[:amax], lw=1, label=f'One-Class SVM (EM-score = {em_auc_ocsvm:.3e})')

plt.ylim([-0.05, 1.05])
plt.xlabel('t')
plt.ylabel('EM(t)')

plt.title('Excess Mass (EM) curves')
plt.legend()

plt.subplot(122)
plt.plot(axis_alpha, mv_iforest, lw=1, label=f'Isolation Forest (AUC = {mv_auc_iforest:.3f})')
plt.plot(axis_alpha, mv_lof, lw=1, label=f'Local Outlier Factor (AUC = {mv_auc_lof:.3f})')
plt.plot(axis_alpha, mv_ocsvm, lw=1, label=f'One-Class SVM (AUC = {mv_auc_ocsvm:.3f})')

plt.xlabel('alpha')
plt.ylabel('MV(alpha)')

plt.title('Mass-Volume (MV) curves')
plt.legend()

plt.savefig("img/eval/em_mv_low_dim_example.png")
plt.show()
