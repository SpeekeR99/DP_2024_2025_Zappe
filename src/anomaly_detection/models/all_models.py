from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

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
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "Isolation Forest", "IF", data_numpy, time_idx, indcs, y_pred_if, anomaly_proba_if, WANTED_FEATURES[1:])
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "One-Class SVM", "OCSVM", data_numpy, time_idx, indcs, y_pred_ocsvm, anomaly_proba_ocsvm, WANTED_FEATURES[1:])
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "Local Outlier Factor", "LOF", data_numpy, time_idx, indcs, y_pred_lof, anomaly_proba_lof, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
