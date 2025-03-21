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
    print("Loading the data...")
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # Take smaller subset of the data (for local computer speed purposes)
    # data = data.head(1000)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Initialize the model
    print("Initializing the model...")
    model_lof = LocalOutlierFactor(n_neighbors=32, contamination=0.01, novelty=True)

    # Train the model
    print("Training the model...")
    kfolds = 5

    y_pred_lof, y_scores_lof, anomaly_proba_lof, em_val_lof, mv_val_lof, em_curve_lof, mv_curve_lof, t, axis_alpha, amax = train_model(model_lof, data_numpy, kfolds=kfolds, eval=True)

    # Prepare data for plots
    print("Plotting the results...")
    time_idx = data.columns.get_loc("Time")
    indcs = [data.columns.get_loc(feature) for feature in WANTED_FEATURES[1:]]  # Skip the "Time" column
    model_names = ["Local Outlier Factor"]
    short_model_names = ["LOF"]
    em_vals = [em_val_lof]
    mv_vals = [mv_val_lof]
    em_curves = [em_curve_lof]
    mv_curves = [mv_curve_lof]

    # Plot the evaluation results
    plot_eval_res(DATE, MARKET_SEGMENT_ID, SECURITY_ID, model_names, short_model_names, em_vals, mv_vals, em_curves, mv_curves, t, axis_alpha, amax)

    # Plot the anomalies
    plot_anomalies(DATE, MARKET_SEGMENT_ID, SECURITY_ID, "Local Outlier Factor", "LOF", data_numpy, time_idx, indcs, y_pred_lof, anomaly_proba_lof, WANTED_FEATURES[1:])


if __name__ == "__main__":
    main()
