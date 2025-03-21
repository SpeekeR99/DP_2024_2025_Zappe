import numpy as np
import torch
from torch.utils.data import DataLoader
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
    # Funnily enough, the CNN expects the channels to be the second dimension and sequences to be the third
    # (batch_size, features, seq_len)
    # However, Transformer expects the sequences to be the second dimension and features to be the third
    # (batch_size, seq_len, features)
    # Whenever you see this if-else statement, remember my pain, when I could not find this for a whole day
    if not model.__class__.__name__ == "TransformerAutoencoder":  # (batch_size, features, seq_len)
        max_features = train_loader.dataset.dataset.data.shape[1]  # Number of features
        if len(train_loader.dataset.dataset.data.shape) == 3:  # FFNN does not have sequences at all
            seq_len = train_loader.dataset.dataset.data.shape[2]
    else:  # (batch_size, seq_len, features)
        max_features = train_loader.dataset.dataset.data.shape[2]
        seq_len = train_loader.dataset.dataset.data.shape[1]
    n_generated_orig = n_generated
    if len(train_loader.dataset.dataset.data.shape) == 3:
        n_generated //= seq_len

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
        # If-else statement from hell again
        if not model.__class__.__name__ == "TransformerAutoencoder":  # (batch_size, features, seq_len)
            X_train = train_loader.dataset.dataset.data[:, features]
            X_test = test_loader.dataset.dataset.data[:, features]
        else:  # (batch_size, seq_len, features)
            X_train = train_loader.dataset.dataset.data[:, :, features]
            X_test = test_loader.dataset.dataset.data[:, :, features]

        # Compute the volume of the support
        # The below code works for FFNN, but CNN/Transformer needs sequences
        # So data is not (batch_size, max_features) but (batch_size, max_features, sequence_length)
        if len(X_train.shape) == 2:
            lim_inf = X_test.min(axis=0).values
            lim_sup = X_test.max(axis=0).values
        else:
            # And again ...
            if not model.__class__.__name__ == "TransformerAutoencoder":  # (batch_size, features, seq_len)
                lim_inf = torch.amin(X_test, dim=(0, 2))
                lim_sup = torch.amax(X_test, dim=(0, 2))
            else: # (batch_size, seq_len, features)
                lim_inf = torch.amin(X_test, dim=(0, 1))
                lim_sup = torch.amax(X_test, dim=(0, 1))
        epsilon = 1e-4  # To avoid division by zero
        volume_support = (lim_sup - lim_inf + epsilon).prod().numpy()

        # Compute the time and alpha axis
        t = np.linspace(0, 100 / volume_support, n_generated)
        axis_alpha = np.linspace(alpha_min, alpha_max, len(mv_curve))
        # Generate uniform samples
        if len(X_train.shape) == 2:
            unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, max_features))
        else:  # X_trian.shape[2] == seq_len
            lim_inf_expanded = np.repeat(lim_inf[:, np.newaxis], seq_len, axis=1)
            lim_sup_expanded = np.repeat(lim_sup[:, np.newaxis], seq_len, axis=1)
            # Even here, yes, again ...
            if not model.__class__.__name__ == "TransformerAutoencoder":  # (batch_size, features, seq_len)
                unif = np.random.uniform(lim_inf_expanded, lim_sup_expanded, size=(n_generated, max_features, seq_len))
            else:  # (batch_size, seq_len, features)
                unif = np.random.uniform(lim_inf_expanded.T, lim_sup_expanded.T, size=(n_generated, seq_len, max_features))
        unif = torch.tensor(unif, dtype=torch.float32)

        # Fit the model
        model.fit(DataLoader(X_train, batch_size=train_loader.batch_size), num_epochs=num_epochs, lr=lr)

        # Compute the scores
        s_X = model.decision_function(X_test)
        s_unif = model.decision_function(unif)
        if len(s_X.shape) == 2:
            s_X = s_X.mean(axis=1)
            s_unif = s_unif.mean(axis=1)
        # ----- Possiblity for the future -- batch size approach (might be needed) -------------------------------------
        # batch_size = 32
        # # s_X = model.decision_function(X_test)
        # s_X = np.zeros((len(X_test), max_features))
        # for i in range(0, len(X_test), batch_size):
        #     low = i
        #     high = min(i + batch_size, len(X_test))
        #     s_X[low:high] = model.score_samples(X_test[low:high])
        # # s_unif = model.decision_function(unif)
        # s_unif = np.zeros((n_generated, max_features))
        # for i in range(0, n_generated, batch_size):
        #     low = i
        #     high = min(i + batch_size, n_generated)
        #     s_unif[low:high] = model.score_samples(unif[low:high])
        # if len(s_X.shape) == 2:
        #     s_X = s_X.mean(axis=1)
        #     s_unif = s_unif.mean(axis=1)
        # ----- Possiblity for the future -- batch size approach (might be needed) -------------------------------------

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
