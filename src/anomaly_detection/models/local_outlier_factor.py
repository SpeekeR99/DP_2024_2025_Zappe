from sklearn.neighbors import LocalOutlierFactor

from src.anomaly_detection.dataloader import load_data
from src.anomaly_detection.training import train_model
from src.anomaly_detection.visuals import plot_anomallies, plot_eval_res


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
    # data = data.head(1000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Initialize the models
    model_lof = LocalOutlierFactor(n_neighbors=32, contamination=0.01, novelty=True)

    # Train the models
    y_pred_lof, y_scores_lof, anomaly_proba_lof, em_val_lof, mv_val_lof, em_curve_lof, mv_curve_lof, t, axis_alpha, amax = train_model(model_lof, data_numpy, kf=5, eval=True)

    # Prepare data for plots
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    model_names = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
    short_model_names = ["IF", "OCSVM", "LOF"]
    em_vals = [em_val_lof]
    mv_vals = [mv_val_lof]
    em_curves = [em_curve_lof]
    mv_curves = [mv_curve_lof]

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    plot_anomallies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "Local Outlier Factor", "LOF", data_numpy, time_idx, indcs, y_pred_lof, anomaly_proba_lof, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
