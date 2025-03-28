import numpy as np


def transform_ys(y_scores, contamination=0.01, lower_is_better=True):
    """
    Transform the scores to predictions based on expected contamination
    :param y_scores: Y scores
    :param contamination: Contamination
    :param lower_is_better: Lower is better (True for NN, False for Scikit models)
    :return: Y predictions, anomaly probability
    """
    how_many_can_be = len(y_scores) * (1 - contamination)
    y_pred = np.zeros_like(y_scores)

    if lower_is_better:
        y_pred[np.argsort(y_scores)[:int(how_many_can_be)]] = 1
        y_pred[np.argsort(y_scores)[int(how_many_can_be):]] = -1
    else:
        y_pred[np.argsort(y_scores)[::-1][:int(how_many_can_be)]] = 1
        y_pred[np.argsort(y_scores)[::-1][int(how_many_can_be):]] = -1

    y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    if lower_is_better:
        anomaly_proba = y_scores_norm
    else:
        anomaly_proba = 1 - y_scores_norm

    return y_pred, anomaly_proba
