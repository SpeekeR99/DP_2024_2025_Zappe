import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import argparse

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from src.anomaly_detection.data.dataloader import load_data, load_data_reduced_dimensions
from src.anomaly_detection.models.training import train_model
from src.anomaly_detection.data.result_transform import transform_ys
from src.anomaly_detection.data.results_file_io import store_results, load_results
from src.anomaly_detection.analysis.visuals import plot_anomalies, plot_eval_res
from src.anomaly_detection.utils import WANTED_FEATURES, RANDOM_SEED_FOR_REPRODUCIBILITY


def main(config, data_file_info, seed):
    """
    Main function
    :param config: Configuration of the model
    :param data_file_info: Information about the data file
    :param seed: Seed for reproducibility
    """
    # Load the data file information
    DATE = data_file_info["date"]
    MARKET_SEGMENT_ID = data_file_info["market_segment_id"]
    SECURITY_ID = data_file_info["security_id"]

    # Load the config
    model_type = config["model_type"]
    kfolds = config["kfolds"]
    n_estimators = config["n_estimators"]
    max_samples = config["max_samples"]
    max_features = config["max_features"]
    gamma = config["gamma"]
    n_neighbors = config["n_neighbors"]

    # Load the data
    print("Loading the data...")
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # data = load_data_reduced_dimensions(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # Take smaller subset of the data (for local computer speed purposes)
    # data = data.head(1000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Initialize the model
    print("Initializing the model...")
    if model_type == "if":
        model = IsolationForest(contamination=0.01, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, random_state=seed)
    elif model_type == "ocsvm":
        model = OneClassSVM(kernel="rbf", nu=0.01, gamma=gamma)
    elif model_type == "lof":
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.01, novelty=True)

    # Train the model
    print("Training the model...")
    y_scores, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = train_model(model, data_numpy, config, kfolds=kfolds, eval=True)
    # y_scores = train_model(model, data_numpy, config, kfolds=kfolds, eval=False)

    # Dump the raw results to results folder
    store_results(DATE, MARKET_SEGMENT_ID, SECURITY_ID, config, y_scores, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax)

    # # Load results (just for reassurance that the function works and that the results are stored correctly)
    # y_pred, y_scores, anomaly_proba, em_val, mv_val, em_curve, mv_curve, t, axis_alpha, amax = load_results(DATE, MARKET_SEGMENT_ID, SECURITY_ID, config, lower_is_better=False)
    #
    # # Prepare data for plots
    # print("Plotting the results...")
    # time_idx = data.columns.get_loc("Time")
    # indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    # if model_type == "if":
    #     model_names = ["Isolation Forest"]
    #     short_model_names = ["IF"]
    # elif model_type == "ocsvm":
    #     model_names = ["One-Class SVM"]
    #     short_model_names = ["OCSVM"]
    # elif model_type == "lof":
    #     model_names = ["Local Outlier Factor"]
    #     short_model_names = ["LOF"]
    # em_vals = [em_val]
    # mv_vals = [mv_val]
    # em_curves = [em_curve]
    # mv_curves = [mv_curve]
    #
    # # Plot the evaluation results
    # plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)
    #
    # # Plot the anomalies
    # plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names[0], short_model_names[0], data_numpy, time_idx, indcs, y_pred, anomaly_proba, WANTED_FEATURES[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--market_id", type=str, default="XEUR")
    parser.add_argument("--date", type=str, default="20191202")
    parser.add_argument("--market_segment_id", type=str, default="688")
    parser.add_argument("--security_id", type=str, default="4128839")

    parser.add_argument("--model_type", type=str, default="if")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_samples", type=str, default="auto")
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--n_neighbors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_FOR_REPRODUCIBILITY)

    args = parser.parse_args()

    data_file_info = {
        "market_id": args.market_id,
        "date": args.date,
        "market_segment_id": args.market_segment_id,
        "security_id": args.security_id
    }

    config = {
        "model_type": args.model_type,
        "kfolds": args.kfolds,
        "n_estimators": args.n_estimators,
        # Max samples can be string "auto" or float or int ...
        "max_samples": args.max_samples if args.max_samples == "auto" else float(args.max_samples) if "." in args.max_samples else int(args.max_samples),
        "max_features": args.max_features,
        "gamma": args.gamma,
        "n_neighbors": args.n_neighbors
    }

    seed = args.seed

    main(config, data_file_info, seed)