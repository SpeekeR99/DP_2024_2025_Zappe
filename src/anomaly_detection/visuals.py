import os
import matplotlib.pyplot as plt


def plot_eval_res(date, market_segment_id, security_id, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax):
    """
    Plot the evaluation results
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param model_names: Model names (expected list, e.g. ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])
    :param short_model_names: Short model names (expected list, e.g. ["IF", "OCSVM", "LOF"])
    :param em_vals: EM values (expected list)
    :param mv_vals: MV values (expected list)
    :param em_curves: EM curves (expected list)
    :param mv_curves: MV curves (expected list)
    :param t: Time axis
    :param axis_alpha: Alpha axis
    :param amax: Alpha maximum
    """
    # Create the output directory if it does not exist
    if not os.path.exists(f"img/anomaly_detections/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"img/anomaly_detections/{date}_{market_segment_id}_{security_id}")

    # Plot the evaluation results
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"{date}_{market_segment_id}_{security_id}")

    for model_name, em_val, mv_val, em_curve, mv_curve in zip(model_names, em_vals, mv_vals, em_curves, mv_curves):
        plt.subplot(121)
        plt.plot(t[:amax], em_curve[:amax], lw=1, label=f"{model_name} (EM-score = {em_val:.3e})")

        plt.subplot(122)
        plt.plot(axis_alpha, mv_curve, lw=1, label=f"{model_name} (AUC = {mv_val:.3f})")

    plt.subplot(121)
    plt.ylim([-0.05, 1.05])
    plt.xlabel("t")
    plt.ylabel("EM(t)")

    plt.title("Excess Mass (EM) curves")
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.xlabel("alpha")
    plt.ylabel("MV(alpha)")

    plt.title("Mass-Volume (MV) curves")
    plt.legend()
    plt.grid()

    short_model_names = "_".join(short_model_names)
    plt.savefig(f"img/anomaly_detections/{date}_{market_segment_id}_{security_id}/{short_model_names}_EM_MV_eval.png")
    plt.show()


def plot_anomallies(date, market_segment_id, security_id, model_name, short_model_name, data_numpy, time_idx, indcs, y_pred, anomaly_proba, required_features):
    """
    Plot the anomalies
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param model_name: Model name
    :param short_model_name: Short model name
    :param data_numpy: Data in numpy format
    :param time_idx: Index of the "Time" column
    :param indcs: Indices of the columns
    :param y_pred: Predictions
    :param anomaly_proba: Probability of anomalies
    :param required_features: Required features
    """
    # Create the output directory if it does not exist
    if not os.path.exists(f"img/anomaly_detections/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"img/anomaly_detections/{date}_{market_segment_id}_{security_id}")

    # Plot the anomalies for each feature
    for i, index in enumerate(indcs):
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"{model_name}")

        plt.scatter(data_numpy[:, time_idx], data_numpy[:, index], color="green", label="Normal")
        plt.scatter(data_numpy[y_pred == -1, time_idx], data_numpy[y_pred == -1, index], color="red", alpha=anomaly_proba[y_pred == -1], label="Anomaly")

        plt.title(required_features[i])

        plt.legend()
        plt.grid()

        plt.savefig(f"img/anomaly_detections/{date}_{market_segment_id}_{security_id}/{short_model_name}_{required_features[i]}.png")
        plt.show()
