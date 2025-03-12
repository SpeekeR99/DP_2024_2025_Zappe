import sys
import time
import numpy as np
import pandas as pd
import datetime

MARKET_ID = "XEUR"
DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"

# XEUR_20210319_589_5336359_detailed
# DATE = "20210319"
# MARKET_SEGMENT_ID = "589"
# SECURITY_ID = "5336359"

if len(sys.argv) == 5:
    MARKET_ID = sys.argv[1]
    DATE = sys.argv[2]
    MARKET_SEGMENT_ID = sys.argv[3]
    SECURITY_ID = sys.argv[4]

INPUT_FILE_PATH = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster.csv"
OUTPUT_FILE_PATH = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster_augmented.csv"

# Add imbalance index, frequency of incoming messages, cancellations rate, etc. to the CSV
print("Augmenting CSV with extra features")
tic = time.time()


def imbalance_index_vectorized(asks, bids, alpha=0.5, level=3):
    """
    Calculate imbalance index for a given orderbook.
    :param asks: numpy matrix of ask sizes (volumes)
    :param bids: numpy matrix of bid sizes (volumes)
    :param alpha: parameter for imbalance index
    :param level: number of levels to consider
    :return: imbalance index
    """
    assert asks.shape[1] >= level and bids.shape[1] >= level, "Not enough levels in orderbook"
    assert alpha > 0, "Alpha must be positive"
    assert level > 0, "Level must be positive"

    # Calculate imbalance index
    V_bt = np.sum(bids[:, :level] * np.exp(-alpha * np.arange(0, level)), axis=1)
    V_at = np.sum(asks[:, :level] * np.exp(-alpha * np.arange(0, level)), axis=1)
    return (V_bt - V_at) / (V_bt + V_at)


def get_frequency_of_all_incoming_actions(timestamps, time=300):
    """
    Returns the frequency of incoming actions (messages) in the last *time* seconds for all timestamps
    :param timestamps: Array of timestamps (in nano seconds)
    :param time: Time window in seconds (default value is 300)
    :return: List of frequencies of incoming actions (messages) for all timestamps
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Initialize an array to hold the frequencies
    freqs = np.zeros_like(timestamps)

    # Calculate the frequency for each timestamp
    for start in range(len(timestamps)-1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        freqs[start] = start - end

    return freqs / time


def get_cancelations_rate(cancellations, timestamps, time=300):
    """
    Returns the cancellations rate in the last *time* seconds for all timestamps
    :param cancellations: List of cancellations for all timestamps
    :param timestamps: Array of timestamps
    :param time_window: Time window in seconds (default value is 300)
    :return: List of cancellations rate for all timestamps
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Initialize an array to hold the cancellations rate
    cancellation_rate = np.zeros_like(timestamps, dtype=float)

    # Calculate the cancellations rate for each timestamp
    for start in range(len(timestamps)-1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        cancellation_rate[start] = np.sum(cancellations[end:start])

    return cancellation_rate / time


def get_high_quoting_activity(data, timestamps, levels=5, time=1):
    """
    High Quoting Activity
    HQ_{i,s(d)} = max_{t∈s(d)} (|EntryAskSize_{i,t} − EntryBidSize_{i,t}| / AskSize_{i,t} + BidSize_{i,t})
    where: EntryAskSizei,t is the increase in the aggregate volume of the orders resting on the
           top 5 ask levels of security i at time t (equal to 0 if there is no increase in the
           aggregate volume)
           EntryBidSizei,t is the increase in the aggregate volume of the orders resting on the
           top 5 bid levels of security i at time t (equal to 0 if there is no increase in the
           aggregate volume)
           AskSizei,t is the cumulative depth (aggregate order quantity) on the top 5 ask levels
           of security i at time t
           BidSizei,t
           is cumulative depth on the top 5 bid levels of security i at time t
           t indexes time (order book events)
           s is a 1-second interval and d is 1-day interval (the metric is calculated for either of
           these frequencies).
    :param data: Dataframe with the orderbook data
    :param timestamps: Array of timestamps
    :param levels: Number of levels to consider (default value is 5)
    :param time: Time window in seconds (default value is 1)
    :return: List of high quoting activities
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    ask_volume_names = [f"Ask Volume {i}" for i in range(1, levels+1)]
    bid_volume_names = [f"Bid Volume {i}" for i in range(1, levels+1)]
    ask_volumes = np.array([data[name].values for name in ask_volume_names])
    bid_volumes = np.array([data[name].values for name in bid_volume_names])
    entry_ask_volumes = ask_volumes[:, 1:] - ask_volumes[:, :-1]
    entry_bid_volumes = bid_volumes[:, 1:] - bid_volumes[:, :-1]
    # Add zeros to the beginning of the arrays (because there's no change and the dimensions must match)
    entry_ask_volumes = np.insert(entry_ask_volumes, 0, 0, axis=1)
    entry_bid_volumes = np.insert(entry_bid_volumes, 0, 0, axis=1)

    # Initialize an array to hold the high quoting activities
    high_quoting_activity = np.zeros_like(timestamps, dtype=float)

    # Calculate the high quoting activity for each timestamp
    for start in range(len(timestamps)-1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        if end < start and start - end > 0:  # Ensure the sliced arrays are non-empty
            ask_bid_sum = ask_volumes[:, end:start] + bid_volumes[:, end:start]
            if ask_bid_sum.size > 0:  # Ensure the sum array is non-empty
                ask_bid_sum[ask_bid_sum == 0] = np.nan  # Avoid division by zero
                high_quoting_activity[start] = np.nanmax(np.abs(entry_ask_volumes[:, end:start] - entry_bid_volumes[:, end:start]) / ask_bid_sum)
            else:
                high_quoting_activity[start] = 0
        else:  # If the sliced arrays are empty, set the high quoting activity to 0 (means no activity in that second)
            high_quoting_activity[start] = 0

    return high_quoting_activity


def get_unbalanced_quoting(data, timestamps, levels=5, time=1):
    """
    Unbalanced Quoting
    UQ_{i,s(d)} = max_{t∈s(d)} (|AskSize_{i,t} − BidSize_{i,t}| / AskSize_{i,t} + BidSize_{i,t})
    where: AskSizei,t is the cumulative depth (aggregate order quantity) on the top 5 ask levels
           of security i at time t
           BidSizei,t
           is cumulative depth on the top 5 bid levels of security i at time t
           t indexes time (order book events)
           s is a 1-second interval and d is 1-day interval (the metric is calculated for either of
           these frequencies).
    :param data: Dataframe with the orderbook data
    :param timestamps: Array of timestamps
    :param levels: Number of levels to consider (default value is 5)
    :param time: Time window in seconds (default value is 1)
    :return: List of high quoting activities
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    ask_volume_names = [f"Ask Volume {i}" for i in range(1, levels+1)]
    bid_volume_names = [f"Bid Volume {i}" for i in range(1, levels+1)]
    ask_volumes = np.array([data[name].values for name in ask_volume_names])
    bid_volumes = np.array([data[name].values for name in bid_volume_names])

    # Initialize an array to hold the high quoting activities
    unbalanced_quoting = np.zeros_like(timestamps, dtype=float)

    # Calculate the high quoting activity for each timestamp
    for start in range(len(timestamps)-1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        if end < start and start - end > 0:  # Ensure the sliced arrays are non-empty
            ask_bid_sum = ask_volumes[:, end:start] + bid_volumes[:, end:start]
            if ask_bid_sum.size > 0:  # Ensure the sum array is non-empty
                ask_bid_sum[ask_bid_sum == 0] = np.nan  # Avoid division by zero
                unbalanced_quoting[start] = np.nanmax(np.abs(ask_volumes[:, end:start] - bid_volumes[:, end:start]) / ask_bid_sum)
            else:
                unbalanced_quoting[start] = 0
        else:  # If the sliced arrays are empty, set the high quoting activity to 0 (means no activity in that second)
            unbalanced_quoting[start] = 0

    return unbalanced_quoting


levels = 30
data = pd.read_csv(INPUT_FILE_PATH, delimiter=",")

ask_columns = [f'Ask Volume {i}' for i in range(1, levels+1)]
bid_columns = [f'Bid Volume {i}' for i in range(1, levels+1)]
timestamps = data["Time"].tolist()

# Imbalance index
lobster_data_matrix = data[ask_columns + bid_columns].values
imbalance_indices = imbalance_index_vectorized(lobster_data_matrix[:, :levels], lobster_data_matrix[:, levels:], alpha=0.5, level=levels)

# Frequency of incoming messages
freqs = get_frequency_of_all_incoming_actions(timestamps, time=300)

# Cancellations rate
cancellations = data["Cancellations"].values
cancellation_rate = get_cancelations_rate(cancellations, timestamps, time=300)

# High Quoting Activity
high_quoting_activity = get_high_quoting_activity(data, timestamps, levels=5, time=1)

# Unbalanced Quoting
unbalanced_quoting = get_unbalanced_quoting(data, timestamps, levels=5, time=1)

# Add the new columns to the CSV
data["Imbalance Index"] = imbalance_indices
data["Frequency of Incoming Messages"] = freqs
data = data.drop(columns=["Cancellations"])  # Drop raw cancellations column
data["Cancellations Rate"] = cancellation_rate
data["High Quoting Activity"] = high_quoting_activity
data["Unbalanced Quoting"] = unbalanced_quoting

# Filter out the timestamps that are not from the correct date
start_nanosec = datetime.datetime.strptime(DATE, "%Y%m%d").timestamp() * 1e9
end_nanosec = start_nanosec + 24 * 60 * 60 * 1e9

timestamps = np.array(data["Time"])
data = data[(timestamps >= start_nanosec) & (timestamps <= end_nanosec)]

# Export to CSV
print("Exporting augmented CSV...")
data.to_csv(OUTPUT_FILE_PATH, index=False)

tac = time.time()
print(f"Augmented CSV in {tac - tic:.2f} seconds")
