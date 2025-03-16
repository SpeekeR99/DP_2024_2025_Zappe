import os
import pandas as pd


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
