import numpy as np
import torch

# https://github.com/ngoix/EMMV_benchmarks.git
from src.anomaly_detection.eval.em import em, mv

from src.anomaly_detection.data.sequences import undo_sequences
from src.anomaly_detection.utils import device


def evaluate(model, data_train, data_test, n_generated=100000, alpha_min=0.9, alpha_max=0.999, t_max=0.9):
    """
    Evaluate the model using the EM and MV scores
    :param model: Model to evaluate
    :param data_train: Training data
    :param data_test: Test data
    :param n_generated: Number of generated samples
    :param alpha_min: Alpha minimum
    :param alpha_max: Alpha maximum
    :param t_max: T maximum
    :return: EM and MV scores, EM and MV curves, time and alpha axis, maximum
    """
    n_features = data_train.shape[1]  # Number of features

    # Compute the volume of the support
    lim_inf = data_test.min(axis=0)
    lim_sup = data_test.max(axis=0)
    epsilon = 1e-4  # To avoid division by zero
    lim_inf = np.nan_to_num(lim_inf, nan=0.0, posinf=0.0, neginf=0.0)
    lim_sup = np.nan_to_num(lim_sup, nan=0.0, posinf=0.0, neginf=0.0)
    volume_support = (lim_sup - lim_inf + epsilon).prod()

    # Compute the time and alpha axis
    t = np.linspace(0, 100 / volume_support, n_generated)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features))  # Generate uniform samples

    # model.fit(data_train)  # Fit the model

    # Compute the scores
    s_X = model.decision_function(data_test)
    s_unif = model.decision_function(unif)

    # Compute the EM
    em_val, em_curve, amax = em(t, t_max, volume_support, s_unif, s_X, n_generated)

    # Compute the MV
    mv_val, mv_curve = mv(axis_alpha, volume_support, s_unif, s_X, n_generated)

    return em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax


def evaluate_torch(model, train_loader, test_loader, y_scores, n_generated=100000, alpha_min=0.9, alpha_max=0.999, t_max=0.9):
    """
    Evaluate the model using the EM and MV scores
    :param model: Model to evaluate
    :param train_loader: Training data
    :param test_loader: Test data
    :param y_scores: Scores of the test data
    :param n_generated: Number of generated samples
    :param alpha_min: Alpha minimum
    :param alpha_max: Alpha maximum
    :param t_max: T maximum
    :return: EM and MV scores, EM and MV curves, time and alpha axis, maximum
    """
    # Max features and sequence length
    n_features = train_loader.dataset.dataset.data.shape[1]  # Number of features
    n_generated_orig = n_generated
    if len(train_loader.dataset.dataset.data.shape) == 3:  # FFNN does not have sequences at all
        seq_len = train_loader.dataset.dataset.data.shape[2]
        n_generated //= seq_len

    X_train = train_loader.dataset.dataset.data
    X_test = test_loader.dataset.dataset.data

    # Compute the volume of the support
    # The below code works for FFNN, but CNN/Transformer needs sequences
    # So data is not (batch_size, n_features) but (batch_size, n_features, sequence_length)
    if len(X_train.shape) == 2:
        lim_inf = X_test.min(axis=0).values
        lim_sup = X_test.max(axis=0).values
    else:
        lim_inf = torch.amin(X_test, dim=(0, 2))
        lim_sup = torch.amax(X_test, dim=(0, 2))
    epsilon = 1e-4  # To avoid division by zero
    lim_inf = np.nan_to_num(lim_inf.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    lim_sup = np.nan_to_num(lim_sup.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    volume_support = (lim_sup - lim_inf + epsilon).prod().cpu().numpy()

    # Compute the time and alpha axis
    t = np.linspace(0, 100 / volume_support, n_generated)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    # Generate uniform samples
    if len(X_train.shape) == 2:
        lim_inf = np.nan_to_num(lim_inf, nan=-epsilon)
        lim_sup = np.nan_to_num(lim_sup, nan=epsilon)
        unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features))
    else:  # X_trian.shape[2] == seq_len
        lim_inf_expanded = np.repeat(lim_inf[:, np.newaxis], seq_len, axis=1)
        lim_sup_expanded = np.repeat(lim_sup[:, np.newaxis], seq_len, axis=1)
        lim_inf_expanded = np.nan_to_num(lim_inf_expanded, nan=-epsilon)
        lim_sup_expanded = np.nan_to_num(lim_sup_expanded, nan=epsilon)
        unif = np.random.uniform(lim_inf_expanded, lim_sup_expanded, size=(n_generated, n_features, seq_len))
    unif = torch.tensor(unif, dtype=torch.float32).to(device)

    # Compute the scores
    s_X = model.decision_function(X_test, contamination=0.01, y_scores=y_scores)
    s_unif = model.decision_function(unif, contamination=0.01)
    if len(s_X.shape) == 2:
        s_X = undo_sequences(s_X, seq_len)
        s_unif = undo_sequences(s_unif, seq_len)

    # Compute the EM
    em_val, em_curve, amax = em(t, t_max, volume_support, s_unif, s_X, n_generated)

    # Compute the MV
    mv_val, mv_curve = mv(axis_alpha, volume_support, s_unif, s_X, n_generated)

    # Interpolate EM curve to the length of the original n_generated
    if n_generated_orig != n_generated:
        t_orig = np.linspace(0, 100 / volume_support, n_generated_orig)
        em_curve = np.interp(t_orig, t, em_curve)

    return em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax
