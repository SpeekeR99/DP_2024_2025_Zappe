import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import argparse

import numpy as np

from src.anomaly_detection.data.dataloader import load_data, load_data_reduced_dimensions
from src.anomaly_detection.data.result_transform import transform_ys
from src.anomaly_detection.data.results_file_io import load_results
from src.anomaly_detection.analysis.visuals import plot_anomalies, plot_eval_res
from src.anomaly_detection.utils import WANTED_FEATURES


def main(config, data_file_info):
    """
    Main function
    :param data_file_info: Information about the data file
    """
    # Load the data file information
    DATE = data_file_info["date"]
    MARKET_SEGMENT_ID = data_file_info["market_segment_id"]
    SECURITY_ID = data_file_info["security_id"]

    # Load the data
    print("Loading the data...")
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    features = WANTED_FEATURES
    # data = load_data_reduced_dimensions(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # features = data.columns.tolist()
    # Take smaller subset of the data (for local computer speed purposes)
    # data = data.head(1000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Initialize the models
    print("Loading the results...")
    config_if = {"model_type": "if", "kfolds": config["kfolds"], "n_estimators": config["if"]["n_estimators"], "max_samples": config["if"]["max_samples"], "max_features": config["if"]["max_features"], "gamma": config["ocsvm"]["gamma"], "n_neighbors": config["lof"]["n_neighbors"]}
    config_ocsvm = {"model_type": "ocsvm", "kfolds": config["kfolds"], "n_estimators": config["if"]["n_estimators"], "max_samples": config["if"]["max_samples"], "max_features": config["if"]["max_features"], "gamma": config["ocsvm"]["gamma"], "n_neighbors": config["lof"]["n_neighbors"]}
    config_lof = {"model_type": "lof", "kfolds": config["kfolds"], "n_estimators": config["if"]["n_estimators"], "max_samples": config["if"]["max_samples"], "max_features": config["if"]["max_features"], "gamma": config["ocsvm"]["gamma"], "n_neighbors": config["lof"]["n_neighbors"]}
    config_ffnn = {"model_type": "ffnn", "epochs": config["epochs"], "kfolds": config["kfolds"], "batch_size": config["ffnn"]["batch_size"], "lr": config["ffnn"]["lr"], "seq_len": config["cnn"]["seq_len"], "latent_dim": config["ffnn"]["latent_dim"]}
    config_cnn = {"model_type": "cnn", "epochs": config["epochs"], "kfolds": config["kfolds"], "batch_size": config["cnn"]["batch_size"], "lr": config["cnn"]["lr"], "seq_len": config["cnn"]["seq_len"], "latent_dim": config["cnn"]["latent_dim"]}
    config_transformer = {"model_type": "transformer", "epochs": config["epochs"], "kfolds": config["kfolds"], "batch_size": config["transformer"]["batch_size"], "lr": config["transformer"]["lr"], "seq_len": config["transformer"]["seq_len"], "latent_dim": config["cnn"]["latent_dim"]}

    # Based on use_model create final configs
    configs_temp = [config_if, config_ocsvm, config_lof, config_ffnn, config_cnn, config_transformer]
    configs = []

    model_names_map = {"if": "Isolation Forest", "ocsvm": "One-Class SVM", "lof": "Local Outlier Factor", "ffnn": "FFNN Autoencoder", "cnn": "CNN Autoencoder", "transformer": "Transformer Autoencoder"}
    model_names = []
    short_model_names_map = {"if": "IF", "ocsvm": "OCSVM", "lof": "LOF", "ffnn": "FFNNAE", "cnn": "CNNAE", "transformer": "TAE"}
    short_model_names = []

    for config_temp in configs_temp:
        if config[config_temp["model_type"]]["use_model"]:
            configs.append(config_temp)

            model_names.append(model_names_map[config_temp["model_type"]])
            short_model_names.append(short_model_names_map[config_temp["model_type"]])

    # Load results (just for reassurance that the function works and that the results are stored correctly)
    y_preds = []
    y_scores = []
    anomaly_probas = []
    em_vals = []
    mv_vals = []
    em_curves = []
    mv_curves = []
    ts = []
    axis_alphas = []
    amaxes = []
    for config in configs:
        y_pred, y_score, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = load_results(DATE, MARKET_SEGMENT_ID, SECURITY_ID, config)
        if len(y_score) != len(data_numpy):
            print(f"Warning: The length of the scores ({len(y_score)}) does not match the length of the data ({len(data_numpy)}).")
            y_pred = y_pred[:len(data_numpy)]
            y_score = y_score[:len(data_numpy)]
            anomaly_proba = anomaly_proba[:len(data_numpy)]
        y_preds.append(y_pred)
        y_scores.append(y_score)
        anomaly_probas.append(anomaly_proba)
        em_vals.append(em_val)
        mv_vals.append(mv_val)
        em_curves.append(em_curve)
        mv_curves.append(mv_curve)
        ts.append(t)
        axis_alphas.append(axis_alpha)
        amaxes.append(amax)

    # Prepare data for plots
    print("Plotting the results...")
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in features[1:]]  # Skip the "Time" column
    t = ts[0]
    axis_alpha = axis_alphas[0]
    amax = max(amaxes)

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    for model_name, short_model_name, y_pred, anomaly_proba in zip(model_names, short_model_names, y_preds, anomaly_probas):
        plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_name, short_model_name, data_numpy, time_idx, indcs, y_pred, anomaly_proba, features[1:])

    # Ensemble -- average model of picked models
    model_name = f"Ensemble ({', '.join(model_names)})"
    short_model_name = f"Ensemble_{'_'.join(short_model_names)}"

    # Scikit Models score -- lower is anomaly, but we want higher to be anomaly
    for i in range(len(y_scores)):
        if short_model_names[i] == "IF" or short_model_names[i] == "OCSVM" or short_model_names[i] == "LOF":
            y_scores[i] = -y_scores[i]
        # Normalize the scores so we can take a "meaningful" average of them
        y_scores[i] = (y_scores[i] - y_scores[i].min()) / (y_scores[i].max() - y_scores[i].min())

    # Create the ensemble model by averaging the normalized scores
    y_scores_ensemble = np.mean(y_scores, axis=0)
    # # ------- Weighted average ---------------------------------------------------------------------------------------
    # # This code is unused, because the weighted average results are very similar to the normal average
    # # Create the ensemble model by weighted averaging the normalized scores
    # # Rank the models based on their EM and MV values separately
    # em_rankings = np.argsort(em_vals)[::-1]  # Best EM is high
    # mv_rankings = np.argsort(mv_vals)  # Best MV is low
    # points = {}
    # for i, model_name in enumerate(model_names):
    #     points[model_name] = 0
    #     points[model_name] += (len(model_names) - em_rankings[i] + len(model_names) - mv_rankings[i]) / 2.0
    # # Normalize the points
    # points = {k: v / sum(points.values()) for k, v in points.items()}
    # # Create the ensemble model by weighted averaging the normalized scores
    # y_scores_ensemble = np.zeros_like(y_scores[0])
    # for i, model_name in enumerate(model_names):
    #     y_scores_ensemble += points[model_name] * y_scores[i]
    # # ------- Weighted average ---------------------------------------------------------------------------------------
    y_pred_ensemble, anomaly_proba_ensemble = transform_ys(y_scores_ensemble, contamination=0.01, lower_is_better=True)

    # # ------- Thresholding ------------------------------------------------------------------------------------------
    # # Threshold the ensemble model to only keep the MOST sure predictions of anomalies
    # # Keep only the predictions, that are 75 % sure or more
    # model_name = f"Ensemble ({', '.join(model_names)}) (only top predictions)"
    # short_model_name = f"Ensemble_top_{'_'.join(short_model_names)}"
    # threshold = 0.75
    # y_pred_ensemble[anomaly_proba_ensemble < threshold] = 1
    # anomaly_proba_ensemble[anomaly_proba_ensemble < threshold] = 0
    # # ------- Thresholding -------------------------------------------------------------------------------------------

    # Plot the anomalies
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_name, short_model_name, data_numpy, time_idx, indcs, y_pred_ensemble, anomaly_proba_ensemble, features[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--market_id", type=str, default="XEUR")
    parser.add_argument("--date", type=str, default="20191202")
    parser.add_argument("--market_segment_id", type=str, default="688")
    parser.add_argument("--security_id", type=str, default="4128839")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--kfolds", type=int, default=5)

    parser.add_argument("--if_n_estimators", type=int, default=100)
    parser.add_argument("--if_max_samples", type=str, default="0.1")
    parser.add_argument("--if_max_features", type=float, default=0.5)
    parser.add_argument("--no_if", type=str, default="false")

    parser.add_argument("--ocsvm_gamma", type=str, default="scale")
    parser.add_argument("--no_ocsvm", type=str, default="false")

    parser.add_argument("--lof_n_neighbors", type=int, default=32)
    parser.add_argument("--no_lof", type=str, default="false")

    parser.add_argument("--ffnn_batch_size", type=int, default=32)
    parser.add_argument("--ffnn_lr", type=float, default=1e-3)
    parser.add_argument("--ffnn_latent_dim", type=int, default=8)
    parser.add_argument("--no_ffnn", type=str, default="false")

    parser.add_argument("--cnn_batch_size", type=int, default=32)
    parser.add_argument("--cnn_lr", type=float, default=1e-3)
    parser.add_argument("--cnn_latent_dim", type=int, default=8)
    parser.add_argument("--cnn_seq_len", type=int, default=64)
    parser.add_argument("--no_cnn", type=str, default="false")

    parser.add_argument("--transformer_batch_size", type=int, default=64)
    parser.add_argument("--transformer_lr", type=float, default=1e-4)
    parser.add_argument("--transformer_seq_len", type=int, default=32)
    parser.add_argument("--no_transformer", type=str, default="false")

    args = parser.parse_args()

    data_file_info = {
        "market_id": args.market_id,
        "date": args.date,
        "market_segment_id": args.market_segment_id,
        "security_id": args.security_id
    }

    config = {
        "epochs": args.epochs,
        "kfolds": args.kfolds,
        "if": {
            "use_model": args.no_if == "false",
            "n_estimators": args.if_n_estimators,
            "max_samples": args.if_max_samples if args.if_max_samples == "auto" else float(args.if_max_samples) if "." in args.if_max_samples else int(args.if_max_samples),
            "max_features": args.if_max_features
        },
        "ocsvm": {
            "use_model": args.no_ocsvm == "false",
            "gamma": args.ocsvm_gamma
        },
        "lof": {
            "use_model": args.no_lof == "false",
            "n_neighbors": args.lof_n_neighbors
        },
        "ffnn": {
            "use_model": args.no_ffnn == "false",
            "batch_size": args.ffnn_batch_size,
            "lr": args.ffnn_lr,
            "latent_dim": args.ffnn_latent_dim
        },
        "cnn": {
            "use_model": args.no_cnn == "false",
            "batch_size": args.cnn_batch_size,
            "lr": args.cnn_lr,
            "latent_dim": args.cnn_latent_dim,
            "seq_len": args.cnn_seq_len
        },
        "transformer": {
            "use_model": args.no_transformer == "false",
            "batch_size": args.transformer_batch_size,
            "lr": args.transformer_lr,
            "seq_len": args.transformer_seq_len
        }
    }

    main(config, data_file_info)
