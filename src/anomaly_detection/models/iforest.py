from sklearn.ensemble import IsolationForest

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
    model_if = IsolationForest(contamination=0.01)
    # model = IsolationForest(contamination=0.01, n_estimators=1000, max_samples=1.0, max_features=1.0)

    # Train the models
    y_pred_if, y_scores_if, anomaly_proba_if, em_val_if, mv_val_if, em_curve_if, mv_curve_if, t, axis_alpha, amax = train_model(model_if, data_numpy, kf=5, eval=True)

    # Prepare data for plots
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    model_names = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
    short_model_names = ["IF", "OCSVM", "LOF"]
    em_vals = [em_val_if]
    mv_vals = [mv_val_if]
    em_curves = [em_curve_if]
    mv_curves = [mv_curve_if]

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    plot_anomallies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "Isolation Forest", "IF", data_numpy, time_idx, indcs, y_pred_if, anomaly_proba_if, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
