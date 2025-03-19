from sklearn.svm import OneClassSVM

from src.anomaly_detection.dataloader import load_data
from src.anomaly_detection.training import train_model
from src.anomaly_detection.visuals import plot_anomalies, plot_eval_res
from src.anomaly_detection.utils import DATE, MARKET_SEGMENT_ID, SECURITY_ID, WANTED_FEATURES


def main():
    """
    Main function
    """
    # Load the data
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # Take smaller subset of the data (for local computer speed purposes)
    data = data.head(10000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Initialize the models
    model_ocsvm = OneClassSVM(kernel="rbf", gamma="scale")

    # Train the models
    kfolds = 5
    y_pred_ocsvm, y_scores_ocsvm, anomaly_proba_ocsvm, em_val_ocsvm, mv_val_ocsvm, em_curve_ocsvm, mv_curve_ocsvm, t, axis_alpha, amax = train_model(model_ocsvm, data_numpy, kfolds=kfolds, eval=True)

    # Prepare data for plots
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    model_names = ["One-Class SVM"]
    short_model_names = ["OCSVM"]
    em_vals = [em_val_ocsvm]
    mv_vals = [mv_val_ocsvm]
    em_curves = [em_curve_ocsvm]
    mv_curves = [mv_curve_ocsvm]

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "One-Class SVM", "OCSVM", data_numpy, time_idx, indcs, y_pred_ocsvm, anomaly_proba_ocsvm, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
