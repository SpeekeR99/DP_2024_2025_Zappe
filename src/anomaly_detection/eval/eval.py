import numpy as np
from sklearn.utils import shuffle
# https://github.com/ngoix/EMMV_benchmarks.git
from lib.eval.em import em, mv

ocsvm_max_train = 10000


def evaluate(model, data_train, data_test, averaging=50, n_generated=100000, alpha_min=0.9, alpha_max=0.999, t_max=0.9):
    """
    Evaluate the model using the EM and MV scores
    :param model: Model to evaluate
    :param data_train: Training data
    :param data_test: Test data
    :param averaging: Averaging iterations
    :param n_generated: Number of generated samples
    :param alpha_min: Alpha minimum
    :param alpha_max: Alpha maximum
    :param t_max: T maximum
    :return: EM and MV scores, EM and MV curves, time and alpha axis, maximum
    """
    max_features = data_train.shape[1]  # Number of features

    # Initialize EM and MV accumulators
    em_val, mv_val = 0, 0
    em_curve = np.zeros(n_generated)
    mv_curve = np.zeros(int((alpha_max - alpha_min) / 0.001))
    # Initialize the maximum
    amax = -1

    # Loop over the averaging
    for _ in range(averaging):
        # Randomly select the features
        features = shuffle(np.arange(max_features))[:max_features]
        X_train = data_train[:, features]
        X_test = data_test[:, features]

        # Compute the volume of the support
        lim_inf = X_test.min(axis=0)
        lim_sup = X_test.max(axis=0)
        epsilon = 1e-4  # To avoid division by zero
        volume_support = (lim_sup - lim_inf + epsilon).prod()

        # Compute the time and alpha axis
        t = np.linspace(0, 100 / volume_support, n_generated)
        axis_alpha = np.linspace(alpha_min, alpha_max, len(mv_curve))
        unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, max_features))  # Generate uniform samples

        # Fit the model
        if model.__class__.__name__ == "OneClassSVM":
            model.fit(X_train[:min(ocsvm_max_train, len(X_train) - 1)])
        else:
            model.fit(X_train)

        # Compute the scores
        s_X = model.decision_function(X_test)
        s_unif = model.decision_function(unif)

        # Compute the EM
        em_val_new, em_curve_new, amax_new = em(t, t_max, volume_support, s_unif, s_X, n_generated)
        amax = max(amax, amax_new)  # Update the maximum

        # Compute the MV
        mv_val_new, mv_curve_new = mv(axis_alpha, volume_support, s_unif, s_X, n_generated)

        # Accumulate results for averaging
        em_val += em_val_new
        mv_val += mv_val_new
        em_curve += em_curve_new
        mv_curve += mv_curve_new

    # Average the results
    em_val /= averaging
    mv_val /= averaging
    em_curve /= averaging
    mv_curve /= averaging

    return em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax