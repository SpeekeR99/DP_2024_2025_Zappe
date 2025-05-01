import os
import sys
import json
import numpy as np

from src.anomaly_detection.data.result_transform import transform_ys


def store_results(date, market_segment_id, security_id, config, y_scores, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax):
    """
    Store the results
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param config: Configuration of the model
    :param y_pred: Predictions
    :param y_scores: Anomaly scores
    :param anomaly_proba: Anomaly probabilities
    :param em_val: EM final value
    :param mv_val: MV final value
    :param em_curve: EM curve
    :param mv_curve: MV curve
    :param t: Time axis
    :param axis_alpha: Alpha axis
    :param amax: Alpha maximum
    """
    # Create the output directory if it does not exist
    if not os.path.exists(f"res/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"res/{date}_{market_segment_id}_{security_id}")

    # Prepare data for storage
    config_string = "_".join([f"{k}={v}" for k, v in config.items()])
    store = {
        "y_scores": y_scores.tolist(),
        "em_val": float(em_val),
        "mv_val": float(mv_val),
        "em_curve": em_curve.tolist(),
        "mv_curve": mv_curve.tolist(),
        "t": t.tolist(),
        "axis_alpha": axis_alpha.tolist(),
        "amax": int(amax)
    }

    # Dump the results
    with open(f"res/{date}_{market_segment_id}_{security_id}/{config_string}.json", "w") as fp:
        json.dump(store, fp)


def load_results(date, market_segment_id, security_id, config, lower_is_better=True):
    """
    Load the results
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param config: Configuration of the model
    :param lower_is_better: Whether lower is better (for anomaly detection)
    :return: y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax
    """
    # Check if the results exist
    config_string = "_".join([f"{k}={v}" for k, v in config.items()])
    if not os.path.exists(f"res/{date}_{market_segment_id}_{security_id}/{config_string}.json"):
        print(f"Results for {date}_{market_segment_id}_{security_id}/{config_string}.json do not exist!")
        sys.exit(1)

    # Prepare string for loading the results
    config_string = "_".join([f"{k}={v}" for k, v in config.items()])

    # Load the results
    with open(f"res/{date}_{market_segment_id}_{security_id}/{config_string}.json", "r") as fp:
        store = json.load(fp)

    # Prepare the results into correct formats
    y_scores = np.array(store["y_scores"])
    em_val = store["em_val"]
    mv_val = store["mv_val"]
    em_curve = np.array(store["em_curve"])
    mv_curve = np.array(store["mv_curve"])
    t = np.array(store["t"])
    axis_alpha = np.array(store["axis_alpha"])
    amax = store["amax"]

    y_pred, anomaly_proba = transform_ys(y_scores, contamination=0.01, lower_is_better=lower_is_better)

    return y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax