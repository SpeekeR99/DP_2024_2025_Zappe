import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def load_data_reduced_dimensions(date="20191202", market_segment_id="688", security_id="4128839", relevant_features=None):
    """
    Load the data
    :param date: Date of the data
    :param market_segment_id: Market segment ID
    :param security_id: Security ID
    :param relevant_features: Relevant features
    :return: Data
    """
    # Load the data
    data = load_data(date, market_segment_id, security_id, relevant_features).dropna()

    # Reduce the dimensions based on correlation and feature importance analysis
    # Keep "Time" of course
    # Combine "Ask Price 1" and "Bid Price 1" into one "Mid Price" using mean of the two
    # Combine "Imbalance Index", "Bid Volume 1" and "Ask Volume 1" into one component using PCA
    # Throw away "Frequency of Incoming Messages" and only keep "Cancellations Rate"
    # Combine "High Quoting Activity", "Unbalanced Quoting", and "Low Execution Probability" into one component using PCA
    # Combine "Trades Oppose Quotes" and "Cancels Oppose Trades" into one component using PCA
    data_dim_reduced = data[["Time"]].copy()

    mid_price = (data["Ask Price 1"] + data["Bid Price 1"]) / 2.0
    data_dim_reduced["Mid Price"] = mid_price

    pca = PCA(n_components=1)
    imbal_data = data[["Imbalance Index", "Bid Volume 1", "Ask Volume 1"]].copy()
    imbal_volumes_pc1 = pca.fit_transform(StandardScaler().fit_transform(imbal_data))
    data_dim_reduced["Imbalance Volumes PC1"] = imbal_volumes_pc1

    data_dim_reduced["Cancellations Rate"] = data["Cancellations Rate"]

    pca = PCA(n_components=1)
    quoting_data = data[["High Quoting Activity", "Unbalanced Quoting", "Low Execution Probability"]].copy()
    quoting_pc1 = pca.fit_transform(StandardScaler().fit_transform(quoting_data))
    data_dim_reduced["Quoting PC1"] = quoting_pc1

    pca = PCA(n_components=1)
    oppose_data = data[["Trades Oppose Quotes", "Cancels Oppose Trades"]].copy()
    oppose_pc1 = pca.fit_transform(StandardScaler().fit_transform(oppose_data))
    data_dim_reduced["Oppose PC1"] = oppose_pc1

    return data_dim_reduced
