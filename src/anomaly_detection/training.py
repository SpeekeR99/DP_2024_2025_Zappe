import numpy as np
from sklearn.model_selection import KFold

from src.anomaly_detection.eval.eval import evaluate, ocsvm_max_train


def train_model(model, data, kf=5, eval=True):
    """
    Train the model
    :param model: Model to train
    :param data: Data
    :param kf: Number of splits in KFold
    :return: Predictions, scores, probability of anomalies
             if eval is True, EM and MV values, EM and MV curves, time and alpha axis, maximum
    """
    # Prepare the data for KFold
    kf = KFold(n_splits=kf)
    y_pred = np.zeros_like(data[:, 0])
    y_scores = np.zeros_like(data[:, 0])

    # Prepare the evaluation results
    em_vals = []
    mv_vals = []
    em_curves = []
    mv_curves = []
    t = -1
    axis_alpha = -1
    amax = -1

    iter = 1
    for train_index, test_index in kf.split(data):
        print(f"KFold {iter}")
        iter += 1

        # Fit the model
        if model.__class__.__name__ == "OneClassSVM":
            model.fit(data[train_index][:min(ocsvm_max_train, len(data[train_index]) - 1)])
        else:
            model.fit(data[train_index])

        # Predict the anomalies in the data
        y_pred[test_index] = model.predict(data[test_index])
        y_scores[test_index] = model.score_samples(data[test_index])

        if eval:
            # Evaluate the model
            em_val, mv_val, em_curve, mv_curve, t_, axis_alpha_, amax_ = evaluate(model, data[train_index], data[test_index])
            em_vals.append(em_val)
            mv_vals.append(mv_val)
            em_curves.append(em_curve)
            mv_curves.append(mv_curve)
            t = t_
            axis_alpha = axis_alpha_
            amax = max(amax, amax_)

    # Calculate the anomaly probability
    y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    anomaly_proba = 1 - y_scores_norm  # The lower the original score, the higher "certainty" it is an anomaly

    # Average the evaluation results
    if eval:
        em_val = np.mean(em_vals)
        mv_val = np.mean(mv_vals)
        em_curve = np.mean(em_curves, axis=0)
        mv_curve = np.mean(mv_curves, axis=0)

        return y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax

    return y_pred, y_scores, anomaly_proba
