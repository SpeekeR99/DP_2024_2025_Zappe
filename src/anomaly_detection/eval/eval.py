import numpy as np
import torch
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
    for iter_avg in range(averaging):
        print(f"Iteration of averaging {iter_avg + 1} / {averaging}")

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


def evaluate_torch(model, train_loader, test_loader, num_epochs=10, lr=1e-5, averaging=50, n_generated=100000, alpha_min=0.9, alpha_max=0.999, t_max=0.9):
    """
    Evaluate the model using the EM and MV scores
    :param model: Model to evaluate
    :param train_loader: Training data
    :param test_loader: Test data
    :param num_epochs: Number of epochs
    :param lr: Learning rate
    :param averaging: Averaging iterations
    :param n_generated: Number of generated samples
    :param alpha_min: Alpha minimum
    :param alpha_max: Alpha maximum
    :param t_max: T maximum
    :return: EM and MV scores, EM and MV curves, time and alpha axis, maximum
    """
    max_features = train_loader.dataset.dataset.data.shape[1]  # Number of features
    n_generated_orig = n_generated
    if len(train_loader.dataset.dataset.data.shape) == 3:
        n_generated //= train_loader.dataset.dataset.data.shape[2]

    # Initialize EM and MV accumulators
    em_val, mv_val = 0, 0
    em_curve = np.zeros(n_generated)
    mv_curve = np.zeros(int((alpha_max - alpha_min) / 0.001))
    # Initialize the maximum
    amax = -1

    # Loop over the averaging
    for iter_avg in range(averaging):
        print(f"Iteration of averaging {iter_avg + 1} / {averaging}")

        # Randomly select the features
        features = shuffle(np.arange(max_features))[:max_features]
        X_train = train_loader.dataset.dataset.data[:, features]
        X_test = test_loader.dataset.dataset.data[:, features]

        # Compute the volume of the support
        # The below code works for FFNN, but CNN/Transformer needs sequences
        # So data is not (batch_size, max_features) but (batch_size, max_features, sequence_length)
        if len(X_train.shape) == 2:
            lim_inf = X_test.min(axis=0).values
            lim_sup = X_test.max(axis=0).values
        else:
            lim_inf = torch.amin(X_test, dim=(0, 2))
            lim_sup = torch.amax(X_test, dim=(0, 2))
        epsilon = 1e-4  # To avoid division by zero
        volume_support = (lim_sup - lim_inf + epsilon).prod().numpy()

        # Compute the time and alpha axis
        t = np.linspace(0, 100 / volume_support, n_generated)
        axis_alpha = np.linspace(alpha_min, alpha_max, len(mv_curve))
        # Generate uniform samples
        if len(X_train.shape) == 2:
            unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, max_features))
        else:  # X_trian.shape[2] == seq_len
            lim_inf_expanded = np.repeat(lim_inf[:, np.newaxis], X_train.shape[2], axis=1)
            lim_sup_expanded = np.repeat(lim_sup[:, np.newaxis], X_train.shape[2], axis=1)
            unif = np.random.uniform(lim_inf_expanded, lim_sup_expanded, size=(n_generated, max_features, X_train.shape[2]))
        unif = torch.tensor(unif, dtype=torch.float32)

        # Fit the model
        model.fit(X_train, num_epochs=num_epochs, lr=lr)

        # Compute the scores
        s_X = model.decision_function(X_test)
        s_unif = model.decision_function(unif)
        if len(s_X.shape) == 2:
            s_X = s_X.mean(axis=1)
            s_unif = s_unif.mean(axis=1)

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

    # Interpolate EM curve to the length of the original n_generated
    if n_generated_orig != n_generated:
        t_orig = np.linspace(0, 100 / volume_support, n_generated_orig)
        em_curve = np.interp(t_orig, t, em_curve)

    return em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax
