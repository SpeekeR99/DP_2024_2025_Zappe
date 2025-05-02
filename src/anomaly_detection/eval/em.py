# ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
# | This code is basically just a copy of the em.py file from the https://github.com/ngoix/EMMV_benchmarks.git         |
# | repository. The only change is in the print statement! Python 2.x -> Python 3.x                                    |
# | I am not the author of this code!                                                                                  |
# |                                                                                                    - Dominik Zappe |
# └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

import numpy as np
from sklearn.metrics import auc
# https://github.com/ngoix/EMMV_benchmarks.git
# This is basically a copy of the em.py file from the above repository
# The only change is in the print statement! Python 2.x -> Python 3.x
# The original print statement was:     print '\n failed to achieve t_max \n'
# The new print statement is:           print('\n failed to achieve t_max \n')
# The following code is not mine, see the submodule eval or https://github.com/ngoix/EMMV_benchmarks.git


def em(t, t_max, volume_support, s_unif, s_X, n_generated):
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.
    for u in s_X_unique:
        # if (s_unif >= u).sum() > n_generated / 1000:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                          t * (s_unif > u).sum() / n_generated
                          * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        print('\n failed to achieve t_max \n')
        amax = -1
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


def mv(axis_alpha, volume_support, s_unif, s_X, n_generated):
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        # pdb.set_trace()
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return auc(axis_alpha, mv), mv
