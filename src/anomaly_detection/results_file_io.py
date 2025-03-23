import os
import sys
import json
import numpy as np


def store_results(date, market_segment_id, security_id, config, y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax):
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
        "y_pred": y_pred.tolist(),
        "y_scores": y_scores.tolist(),
        "anomaly_proba": anomaly_proba.tolist(),
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


def load_results(date, market_segment_id, security_id, config):
    """
    Load the results
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param config: Configuration of the model
    :return:
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
    y_pred = np.array(store["y_pred"])
    y_scores = np.array(store["y_scores"])
    anomaly_proba = np.array(store["anomaly_proba"])
    em_val = store["em_val"]
    mv_val = store["mv_val"]
    em_curve = np.array(store["em_curve"])
    mv_curve = np.array(store["mv_curve"])
    t = np.array(store["t"])
    axis_alpha = np.array(store["axis_alpha"])
    amax = store["amax"]

    return y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax