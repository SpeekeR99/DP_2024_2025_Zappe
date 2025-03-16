import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from src.anomaly_detection.eval.eval import evaluate
from src.anomaly_detection.eval.eval import ocsvm_max_train


def load_data(date="20191202", market_segment_id="688", security_id="4128839", relevant_features=None):
    """
    Load the data
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param relevant_features: Relevant features
    :return: Data
    """
    # Assert the input file exists
    filepath = f"data/{date}_{market_segment_id}_{security_id}_lobster_augmented.csv"
    if not os.path.exists(filepath):
        print(f"The input file {filepath} does not exist.")
        exit(1)

    # If the relevant features are not specified, use the default ones
    if relevant_features is None:
        relevant_features = ["Time"]

    # Read the data
    data = pd.read_csv(filepath)

    # Keep only the "relevant" features
    try:
        data = data[relevant_features]
    except KeyError as e:
        print(f"The input file does not contain the necessary column {e}.")
        exit(1)

    return data


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


def main():
    """
    Main function
    """
    # Load the data
    DATE = "20191202"
    MARKET_SEGMENT_ID = "688"
    SECURITY_ID = "4128839"
    WANTED_FEATURES = [
        "Time",
        "Ask Price 1",
        "Ask Volume 1",
        "Bid Price 1",
        "Bid Volume 1",
        "Imbalance Index",
        "Frequency of Incoming Messages",
        "Cancellations Rate",
        "High Quoting Activity",
        "Unbalanced Quoting",
        "Low Execution Probability",
        "Trades Oppose Quotes",
        "Cancels Oppose Trades"
    ]
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # Take smaller subset of the data (for local computer speed purposes)
    data = data.head(1000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Initialize the models
    model_if = IsolationForest(contamination=0.01)
    # model = IsolationForest(contamination=0.01, n_estimators=1000, max_samples=1.0, max_features=1.0)
    model_ocsvm = OneClassSVM(kernel="rbf", gamma="scale")
    model_lof = LocalOutlierFactor(n_neighbors=32, contamination=0.01, novelty=True)

    # Train the models
    y_pred_if, y_scores_if, anomaly_proba_if, em_val_if, mv_val_if, em_curve_if, mv_curve_if, t_if, axis_alpha_if, amax_if = train_model(model_if, data_numpy, kf=5, eval=True)
    y_pred_ocsvm, y_scores_ocsvm, anomaly_proba_ocsvm, em_val_ocsvm, mv_val_ocsvm, em_curve_ocsvm, mv_curve_ocsvm, t_ocsvm, axis_alpha_ocsvm, amax_ocsvm = train_model(model_ocsvm, data_numpy, kf=5, eval=True)
    y_pred_lof, y_scores_lof, anomaly_proba_lof, em_val_lof, mv_val_lof, em_curve_lof, mv_curve_lof, t_lof, axis_alpha_lof, amax_lof = train_model(model_lof, data_numpy, kf=5, eval=True)

    # Prepare data for plots
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    model_names = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
    short_model_names = ["IF", "OCSVM", "LOF"]
    em_vals = [em_val_if, em_val_ocsvm, em_val_lof]
    mv_vals = [mv_val_if, mv_val_ocsvm, mv_val_lof]
    em_curves = [em_curve_if, em_curve_ocsvm, em_curve_lof]
    mv_curves = [mv_curve_if, mv_curve_ocsvm, mv_curve_lof]
    t = t_if
    axis_alpha = axis_alpha_if
    amax = max(amax_if, amax_ocsvm, amax_lof)

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    plot_anomallies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "Isolation Forest", "IF", data_numpy, time_idx, indcs, y_pred_if, anomaly_proba_if, WANTED_FEATURES[1:])
    plot_anomallies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "One-Class SVM", "OCSVM", data_numpy, time_idx, indcs, y_pred_ocsvm, anomaly_proba_ocsvm, WANTED_FEATURES[1:])
    plot_anomallies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "Local Outlier Factor", "LOF", data_numpy, time_idx, indcs, y_pred_lof, anomaly_proba_lof, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
