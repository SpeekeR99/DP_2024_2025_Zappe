import os
import numpy as np
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


def plot_anomalies(date, market_segment_id, security_id, model_name, short_model_name, data_numpy, time_idx, indcs, y_pred, anomaly_proba, required_features):
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


def plot_feat_corr(date, market_segment_id, security_id, corr_matrix):
    """
    Plot the feature correlation
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param corr_matrix: Correlation matrix
    """
    # Create the output directory if it does not exist
    if not os.path.exists(f"img/features/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"img/features/{date}_{market_segment_id}_{security_id}")

    # Plot the correlation matrix
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    # Draw matrix
    mat = ax.matshow(corr_matrix, cmap="bwr")
    mat.set_clim(vmin=-1, vmax=1)
    plt.colorbar(mat)

    # Draw labels
    labels = corr_matrix.index
    for (i, j), z in np.ndenumerate(corr_matrix):
        ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center", fontsize=10)

    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    labels_clipped = [label[:10] for label in labels]
    ax.set_xticklabels(labels_clipped, rotation=45//4)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Correlation")

    plt.savefig(f"img/features/{date}_{market_segment_id}_{security_id}/feature_correlation.png")
    plt.show()


def plot_feat_imp(date, market_segment_id, security_id, feature_importance_dict, wanted_features):
    """
    Plot the feature importance
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param feature_importance_dict: Dictionary of feature importance (key = feature, value = array of importance)
    :param wanted_features: Wanted features
    """
    # Create the output directory if it does not exist
    if not os.path.exists(f"img/features/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"img/features/{date}_{market_segment_id}_{security_id}")

    # Plot the feature importance
    plt.figure(figsize=(20, 10))

    # Boxplot for each feature
    plt.boxplot([feature_importance_dict[feature] for feature in feature_importance_dict])

    plt.xticks(range(1, len(wanted_features) + 1), wanted_features, rotation=45 // 2)

    plt.title("Feature Importance")

    plt.grid()

    plt.savefig(f"img/features/{date}_{market_segment_id}_{security_id}/feature_importance.png")
    plt.show()
