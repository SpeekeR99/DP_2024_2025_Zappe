import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import pandas
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D


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
    if not os.path.exists(f"img/eval/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"img/eval/{date}_{market_segment_id}_{security_id}")

    # Plot the evaluation results
    for i in range(2):  # Plot non-log and log scale versions
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"{date}_{market_segment_id}_{security_id}")

        for model_name, em_val, mv_val, em_curve, mv_curve in zip(model_names, em_vals, mv_vals, em_curves, mv_curves):
            plt.subplot(121)
            plt.plot(t[:amax], em_curve[:amax], lw=1, label=f"{model_name} (EM-score = {em_val:.3e})")

            plt.subplot(122)
            plt.plot(axis_alpha, mv_curve, lw=1, label=f"{model_name} (AUC = {mv_val:.3f})")

        plt.subplot(121)
        plt.ylim([-0.05, 1.05])
        if i != 0:
            plt.xlabel("log(t)")
            plt.xscale("log")
        else:
            plt.xlabel("t")
        plt.ylabel("EM(t)")

        plt.title("Excess Mass (EM) curves")
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.xlabel("alpha")
        if i != 0:
            plt.ylabel("log(MV(alpha))")
            plt.yscale("log")
        else:
            plt.ylabel("MV(alpha)")

        plt.title("Mass-Volume (MV) curves")
        plt.legend()
        plt.grid()

        file_name = "_".join(short_model_names)
        if i != 0:
            plt.savefig(f"img/eval/{date}_{market_segment_id}_{security_id}/{file_name}_EM_MV_eval_log.png")
        else:
            plt.savefig(f"img/eval/{date}_{market_segment_id}_{security_id}/{file_name}_EM_MV_eval.png")
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

    # Convert the time data to datetime
    time_data = np.array([
        (datetime.datetime.min + datetime.timedelta(seconds=t // 1e9))
        for t in data_numpy[:, time_idx]
    ])

    # Get the anomalies
    anomaly_timestamps = np.array([
        (datetime.datetime.min + datetime.timedelta(seconds=t // 1e9))
        for t in data_numpy[y_pred == -1, time_idx]
    ])
    anomaly_alpha = anomaly_proba[y_pred == -1]
    # Normalize the alpha to [0, 1]
    anomaly_alpha = (anomaly_alpha - anomaly_alpha.min()) / (anomaly_alpha.max() - anomaly_alpha.min())
    # Replace NaN values with 0
    anomaly_alpha = np.nan_to_num(anomaly_alpha)

    if len(time_data) > 500_000:
        anomaly_alpha *= 0.1  # Reduce the alpha for better visualization
    if len(time_data) > 1_000_000:
        anomaly_alpha *= 0.5  # Reduce the alpha even more

    # Plot the anomalies for each feature
    for i, index in enumerate(indcs):
        feature = required_features[i]

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"{model_name}")

        for timestamp, anomaly_proba in zip(anomaly_timestamps, anomaly_alpha):
            plt.axvspan(timestamp, timestamp, color="red", alpha=anomaly_proba)
        if "Oppose" in feature:  # Trades Oppose Quotes and Cancels Oppose Trades are categorical (True/False)
            plt.scatter(time_data, data_numpy[:, index], color="black", label="Normal", alpha=0.75)
        else:
            plt.plot(time_data, data_numpy[:, index], color="black", label="Normal", alpha=0.75)

        plt.title(feature)
        plt.xlabel("Time")
        plt.ylabel(feature)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=22.5, ha='right')

        normal_patch = Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=10, label="Normal")
        anomaly_patch = Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markersize=10, label="Anomaly")
        plt.legend(handles=[normal_patch, anomaly_patch])

        plt.grid()

        plt.savefig(f"img/anomaly_detections/{date}_{market_segment_id}_{security_id}/{short_model_name}_{feature}.png")
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
    ax.set_xticklabels(labels_clipped, rotation=22.5, ha='left')
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

    plt.xticks(range(1, len(wanted_features) + 1), wanted_features, rotation=22.5, ha='right')

    plt.title("Feature Importance")

    plt.grid()

    plt.savefig(f"img/features/{date}_{market_segment_id}_{security_id}/feature_importance.png")
    plt.show()


def plot_basic_dimensional_vis(date, market_segment_id, security_id):
    """
    Plot the basic dimensional visualization
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    """
    FILE_PATH = f"data/{date}_{market_segment_id}_{security_id}_lobster_augmented.csv"
    if not os.path.exists(f"img/features/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"img/features/{date}_{market_segment_id}_{security_id}")

    data = pandas.read_csv(FILE_PATH)
    data_numpy = data.to_numpy()
    time_idx = data.columns.get_loc("Time")

    # Convert the time data to datetime
    time_data = np.array([
        (datetime.datetime.min + datetime.timedelta(seconds=t // 1e9))
        for t in data_numpy[:, time_idx]
    ])

    # Visualize all the columns in the dataset with "Time" on the x-axis
    for column in data.columns:
        if column == "Time" or "Ask Price" in column or "Bid Price" in column or "Ask Volume" in column or "Bid Volume" in column:
            if column != "Ask Price 1" and column != "Bid Price 1" and column != "Ask Volume 1" and column != "Bid Volume 1":
                continue
        plt.figure(figsize=(20, 10))
        plt.title(column)

        if "Oppose" in column:  # Trades Oppose Quotes and Cancels Oppose Trades are categorical (True/False)
            plt.scatter(time_data, data[column], color="black", label=column)
        else:
            plt.plot(time_data, data[column], color="black", label=column)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=22.5, ha='right')

        plt.grid()
        plt.savefig(f"img/features/{date}_{market_segment_id}_{security_id}/{column}.png")
        plt.show()


def plot_pareto(date, market_segment_id, security_id, pca):
    """
    Pareto graph for PCA analysis
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param pca: PCA object from sklearn
    """
    # Create the output directory if it does not exist
    if not os.path.exists(f"img/features/{date}_{market_segment_id}_{security_id}"):
        os.makedirs(f"img/features/{date}_{market_segment_id}_{security_id}")

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    # Plot explained variance ratio
    ax.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color="green")

    ax.set_xticks(np.arange(len(pca.explained_variance_ratio_)))
    ax.set_xticklabels(np.arange(1, len(pca.explained_variance_ratio_) + 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1)

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Proportion of variance explained")
    ax.set_title("Pareto graph")

    # Plot cumulative sum
    ax.plot(np.cumsum(pca.explained_variance_ratio_), color="red")

    # Create legend
    bar_patch = mpatches.Patch(color="green", label="Explained Variance")
    line_patch = mlines.Line2D([], [], color="red", label="Cumulative Sum")
    ax.legend(handles=[bar_patch, line_patch], loc="best")

    plt.grid()
    plt.savefig(f"img/features/{date}_{market_segment_id}_{security_id}/PCA_pareto.png")
    plt.show()


def plot_loadings(date, market_segment_id, security_id, pca, df):
    """
    Biplot of Principal Component 1 and Principal Component 2 (scatter) with loadings (arrows)
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param pca: PCA object from sklearn
    :param df: Data (original)
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    # Loadings
    loadings = pca.components_
    features = df.columns
    colors = plt.cm.rainbow(np.linspace(0, 1, len(features)))

    for i, (feature, color) in enumerate(zip(features, colors)):
        # Plot arrows with different colors
        ax.arrow(0, 0, loadings[0, i], loadings[1, i], head_width=0.02, head_length=0.05, fc=color, ec=color)
        # ax.text(loadings[0, i], loadings[1, i], feature, color="k", ha="center", va="center")  # + Jitter
        ax.text(loadings[0, i] + np.random.normal(0, 0.02), loadings[1, i] + np.random.normal(0, 0.02), feature, color="k", ha="center", va="center")

    # Create legend
    patches = [mpatches.Patch(color=color, label=feature) for feature, color in zip(features, colors)]
    ax.legend(handles=patches, loc="best")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Loadings")

    plt.grid()
    plt.savefig(f"img/features/{date}_{market_segment_id}_{security_id}/PCA_loadings.png")
    plt.show()


if __name__ == "__main__":
    # DATE = 20191202
    # MARKET_SEGMENT_ID = 688
    # SECURITY_ID = 4128839
    DATE = "20210319"
    MARKET_SEGMENT_ID = "688"
    SECURITY_ID = "5578483"
    plot_basic_dimensional_vis(DATE, MARKET_SEGMENT_ID, SECURITY_ID)
