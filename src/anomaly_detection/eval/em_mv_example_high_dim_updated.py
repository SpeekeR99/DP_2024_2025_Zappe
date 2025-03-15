import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle as sh
from sklearn.datasets import load_iris
# https://github.com/ngoix/EMMV_benchmarks.git
from lib.eval.em import em, mv

# This example is only an updated version of old example from https://github.com/ngoix/EMMV_benchmarks.git

# Parameters
averaging = 50  # Number of feature sub-sampling iterations
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

max_features = X.shape[1]

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

# Initialize EM and MV accumulators
em_iforest, mv_iforest = 0, 0
em_lof, mv_lof = 0, 0
em_ocsvm, mv_ocsvm = 0, 0

# Storage for averaged curves
em_iforest_curve = np.zeros(n_generated)
em_lof_curve = np.zeros(n_generated)
em_ocsvm_curve = np.zeros(n_generated)

mv_iforest_curve = np.zeros(int((alpha_max - alpha_min) / 0.001))
mv_lof_curve = np.zeros_like(mv_iforest_curve)
mv_ocsvm_curve = np.zeros_like(mv_iforest_curve)

# Storage for the maximums
amax_iforest = -1
amax_lof = -1
amax_ocsvm = -1

nb_exp = 0

while nb_exp < averaging:
    features = sh(np.arange(n_features))[:max_features]  # Randomly select subset of features
    X_train_ = X_train[:, features]
    X_ = X_test[:, features]

    lim_inf = X_.min(axis=0)
    lim_sup = X_.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()

    if volume_support > 0:
        nb_exp += 1
        t = np.linspace(0, 100 / volume_support, n_generated)  # Ensure consistent t values
        axis_alpha = np.linspace(alpha_min, alpha_max, len(mv_iforest_curve))
        unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, max_features))

        iforest.fit(X_train_)
        lof.fit(X_train_)
        ocsvm.fit(X_train_[:min(ocsvm_max_train, n_samples_train - 1)])

        s_X_iforest = iforest.decision_function(X_)
        s_X_lof = lof.decision_function(X_)
        s_X_ocsvm = ocsvm.decision_function(X_).reshape(-1)

        s_unif_iforest = iforest.decision_function(unif)
        s_unif_lof = lof.decision_function(unif)
        s_unif_ocsvm = ocsvm.decision_function(unif).reshape(-1)

        # Compute EM and MV curves (store both AUC and curve)
        em_iforest_val, em_iforest_curve_new, amax_iforest_new = em(t, t_max, volume_support, s_unif_iforest,
                                                                    s_X_iforest, n_generated)
        amax_iforest = max(amax_iforest, amax_iforest_new)
        mv_iforest_val, mv_iforest_curve_new = mv(axis_alpha, volume_support, s_unif_iforest, s_X_iforest, n_generated)

        em_lof_val, em_lof_curve_new, amax_lof_new = em(t, t_max, volume_support, s_unif_lof, s_X_lof, n_generated)
        amax_lof = max(amax_lof, amax_lof_new)
        mv_lof_val, mv_lof_curve_new = mv(axis_alpha, volume_support, s_unif_lof, s_X_lof, n_generated)

        em_ocsvm_val, em_ocsvm_curve_new, amax_ocsvm_new = em(t, t_max, volume_support, s_unif_ocsvm, s_X_ocsvm,
                                                              n_generated)
        amax_ocsvm = max(amax_ocsvm, amax_ocsvm_new)
        mv_ocsvm_val, mv_ocsvm_curve_new = mv(axis_alpha, volume_support, s_unif_ocsvm, s_X_ocsvm, n_generated)

        # Accumulate results for averaging
        em_iforest += em_iforest_val
        mv_iforest += mv_iforest_val
        em_lof += em_lof_val
        mv_lof += mv_lof_val
        em_ocsvm += em_ocsvm_val
        mv_ocsvm += mv_ocsvm_val

        em_iforest_curve += em_iforest_curve_new
        em_lof_curve += em_lof_curve_new
        em_ocsvm_curve += em_ocsvm_curve_new

        mv_iforest_curve += mv_iforest_curve_new
        mv_lof_curve += mv_lof_curve_new
        mv_ocsvm_curve += mv_ocsvm_curve_new

# Compute final averaged curves
em_iforest /= averaging
mv_iforest /= averaging
em_lof /= averaging
mv_lof /= averaging
em_ocsvm /= averaging
mv_ocsvm /= averaging

em_iforest_curve /= averaging
em_lof_curve /= averaging
em_ocsvm_curve /= averaging

mv_iforest_curve /= averaging
mv_lof_curve /= averaging
mv_ocsvm_curve /= averaging

if amax_iforest == -1 or amax_lof == -1 or amax_ocsvm == -1:
    amax = -1
else:
    amax = max(amax_iforest, amax_lof, amax_ocsvm)

# Plot Excess-Mass (EM) Curves
fig = plt.figure(figsize=(15, 10))

plt.subplot(121)
plt.plot(t[:amax], em_iforest_curve[:amax], lw=1, label=f'Isolation Forest (EM-score = {em_iforest:.3e})')
plt.plot(t[:amax], em_lof_curve[:amax], lw=1, label=f'Local Outlier Factor (EM-score = {em_lof:.3e})')
plt.plot(t[:amax], em_ocsvm_curve[:amax], lw=1, label=f'One-Class SVM (EM-score = {em_ocsvm:.3e})')

plt.ylim([-0.05, 1.05])
plt.xlabel('t')
plt.ylabel('EM(t)')

plt.title('Excess Mass (EM) curves')
plt.legend()

plt.subplot(122)
plt.plot(axis_alpha, mv_iforest_curve, lw=1, label=f'Isolation Forest (AUC = {mv_iforest:.3f})')
plt.plot(axis_alpha, mv_lof_curve, lw=1, label=f'Local Outlier Factor (AUC = {mv_lof:.3f})')
plt.plot(axis_alpha, mv_ocsvm_curve, lw=1, label=f'One-Class SVM (AUC = {mv_ocsvm:.3f})')

plt.xlabel('alpha')
plt.ylabel('MV(alpha)')

plt.title('Mass-Volume (MV) curves')
plt.legend()

plt.savefig("img/eval/em_mv_high_dim_example.png")
plt.show()
