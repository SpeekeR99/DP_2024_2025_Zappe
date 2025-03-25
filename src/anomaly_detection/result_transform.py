import numpy as np


def transform_ys(y_scores, contamination=0.01):
    """
    Transform the scores to predictions based on expected contamination
    :param y_scores: Y scores
    :param contamination: Contamination
    :return: Y predictions, anomaly probability
    """
    how_many_can_be = len(y_scores) * contamination
    y_pred = np.zeros_like(y_scores)
    y_pred[np.argsort(y_scores)[:int(how_many_can_be)]] = -1
    y_pred[np.argsort(y_scores)[int(how_many_can_be):]] = 1

    y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    anomaly_proba = 1 - y_scores_norm

    return y_pred, anomaly_proba