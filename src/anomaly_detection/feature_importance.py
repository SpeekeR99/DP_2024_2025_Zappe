import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
# https://github.com/britojr/diffi.git
from lib.diffi.diffi.diffi import diffi_score

from src.anomaly_detection.dataloader import load_data
from src.anomaly_detection.visuals import plot_feat_corr, plot_feat_imp
from src.anomaly_detection.utils import DATE, MARKET_SEGMENT_ID, SECURITY_ID, WANTED_FEATURES


def main():
    """
    Main function
    """
    # Load the data
    data = load_data(date=DATE, market_segment_id=MARKET_SEGMENT_ID, security_id=SECURITY_ID, relevant_features=WANTED_FEATURES)

    # Transform the data to numpy and drop NaN values
    data_numpy = data.dropna().to_numpy()

    # Correlation analysis
    corr_mat = data.corr()
    plot_feat_corr(DATE, MARKET_SEGMENT_ID, SECURITY_ID, corr_mat)

    # Initialize the models
    model = IsolationForest(contamination=0.01)
    # model = IsolationForest(contamination=0.01, n_estimators=1000, max_samples=1.0, max_features=1.0)

    # Check if the feature importance has already been calculated
    if not os.path.exists(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_feature_importance.pkl"):
        # Calculate the feature importance a lot of times (to get a more stable result)
        iter_num = 100
        feature_importance = {}  # Prepare dict for results
        for i in range(len(WANTED_FEATURES)):
            feature_importance[WANTED_FEATURES[i]] = []

        for i in range(iter_num):
            print(i)
            # Fit all the data to determine the Feature Importance
            model.fit(data_numpy)
            feat_imp = diffi_score(model, data_numpy)

            # Save the feature importance
            for j in range(len(feat_imp)):
                feature_importance[WANTED_FEATURES[j]].append(feat_imp[j])

        # Save the feature importance values
        with open(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_feature_importance.pkl", "wb") as fp:
            pickle.dump(feature_importance, fp)
    else:
        # Load the feature importance values
        with open(f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_feature_importance.pkl", "rb") as fp:
            feature_importance = pickle.load(fp)

    # Throw away "crazy" big outliers
    for feature in feature_importance:
        feature_importance[feature] = [x for x in feature_importance[feature] if x < 50.0]

    # Plot the feature importance
    plot_feat_imp(DATE, MARKET_SEGMENT_ID, SECURITY_ID, feature_importance, WANTED_FEATURES)

    # Calculate the mean and std of the feature importance
    for feature in feature_importance:
        print(f"Feature: {feature}")
        print(f"Mean: {np.mean(feature_importance[feature])}")
        print(f"Std: {np.std(feature_importance[feature])}")
        print()


if __name__ == "__main__":
    main()
