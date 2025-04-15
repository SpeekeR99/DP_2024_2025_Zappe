import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import argparse

import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
# https://github.com/britojr/diffi.git
from lib.diffi.diffi.diffi import diffi_score

from src.anomaly_detection.data.dataloader import load_data, load_data_reduced_dimensions
from src.anomaly_detection.analysis.visuals import plot_feat_corr, plot_pareto, plot_loadings, plot_feat_imp
from src.anomaly_detection.utils import WANTED_FEATURES


def main(data_file_info):
    """
    Main function
    :param data_file_info: Information about the data file
    """
    # Load the data file information
    DATE = data_file_info["date"]
    MARKET_SEGMENT_ID = data_file_info["market_segment_id"]
    SECURITY_ID = data_file_info["security_id"]

    # Load the data
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    features = WANTED_FEATURES
    # data = load_data_reduced_dimensions(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)
    # features = data.columns.tolist()

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Correlation analysis
    corr_mat = data.corr()
    plot_feat_corr(DATE, MARKET_SEGMENT_ID, SECURITY_ID, corr_mat)

    # PCA analysis
    scaler = StandardScaler()
    data_numpy_scaled = scaler.fit_transform(data_numpy)

    pca = PCA(n_components=data_numpy_scaled.shape[1])
    pca.fit_transform(data_numpy_scaled)

    # Plot PCA
    plot_pareto(DATE, MARKET_SEGMENT_ID, SECURITY_ID, pca)
    plot_loadings(DATE, MARKET_SEGMENT_ID, SECURITY_ID, pca, data)

    # Initialize the models
    model = IsolationForest(contamination=0.01)
    # model = IsolationForest(contamination=0.01, n_estimators=1000, max_samples=1.0, max_features=1.0)

    # Check if the feature importance has already been calculated
    if not os.path.exists(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_feature_importance.pkl"):
    # if not os.path.exists(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_dim_reduced_feature_importance.pkl"):
        # Calculate the feature importance a lot of times (to get a more stable result)
        iter_num = 100
        feature_importance = {}  # Prepare dict for results
        for i in range(len(features)):
            feature_importance[features[i]] = []

        for i in range(iter_num):
            print(i)
            # Fit all the data to determine the Feature Importance
            model.fit(data_numpy)
            feat_imp = diffi_score(model, data_numpy)

            # Save the feature importance
            for j in range(len(feat_imp)):
                feature_importance[features[j]].append(feat_imp[j])

        # Save the feature importance values
        with open(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_feature_importance.pkl", "wb") as fp:
        # with open(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_dim_reduced_feature_importance.pkl", "wb") as fp:
            pickle.dump(feature_importance, fp)
    else:
        # Load the feature importance values
        with open(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_feature_importance.pkl", "rb") as fp:
        # with open(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_dim_reduced_feature_importance.pkl", "rb") as fp:
            feature_importance = pickle.load(fp)

    # Throw away "crazy" big outliers
    for feature in feature_importance:
        feature_importance[feature] = [x for x in feature_importance[feature] if x < 50.0]

    # Plot the feature importance
    plot_feat_imp(DATE, MARKET_SEGMENT_ID, SECURITY_ID, feature_importance, features)

    # Calculate the mean and std of the feature importance
    for feature in feature_importance:
        print(f"Feature: {feature}")
        print(f"Mean: {np.mean(feature_importance[feature])}")
        print(f"Std: {np.std(feature_importance[feature])}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--market_id", type=str, default="XEUR")
    parser.add_argument("--date", type=str, default="20191202")
    parser.add_argument("--market_segment_id", type=str, default="688")
    parser.add_argument("--security_id", type=str, default="4128839")

    args = parser.parse_args()

    data_file_info = {
        "market_id": args.market_id,
        "date": args.date,
        "market_segment_id": args.market_segment_id,
        "security_id": args.security_id
    }

    main(data_file_info)
