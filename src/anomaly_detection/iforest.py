import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the data
DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"

filepath = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster_augmented.csv"
if not os.path.exists(f"img/anomaly_detections/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}"):
    os.makedirs(f"img/anomaly_detections/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}")
data = pd.read_csv(filepath)

# Keep only the "relevant" features
try:
    data = data[["Time", "Ask Price 1", "Ask Volume 1", "Bid Price 1", "Bid Volume 1", "Imbalance Index", "Frequency of Incoming Messages", "Cancellations Rate", "High Quoting Activity", "Unbalanced Quoting", "Low Execution Probability", "Trades Oppose Quotes", "Cancels Oppose Trades"]]
except KeyError as e:
    print(f"The input file does not contain the necessary column {e}.")
    exit(1)

# Take smaller subset of the data (for local computer speed purposes)
data = data.head(100000)

# Get the indices of the columns
time_idx = data.columns.get_loc("Time")
ask_price_idx = data.columns.get_loc("Ask Price 1")
ask_volume_idx = data.columns.get_loc("Ask Volume 1")
bid_price_idx = data.columns.get_loc("Bid Price 1")
bid_volume_idx = data.columns.get_loc("Bid Volume 1")
imbalance_idx = data.columns.get_loc("Imbalance Index")
frequency_idx = data.columns.get_loc("Frequency of Incoming Messages")
cancellation_rate_idx = data.columns.get_loc("Cancellations Rate")
high_quoting_activity_idx = data.columns.get_loc("High Quoting Activity")
unbalanced_quoting_idx = data.columns.get_loc("Unbalanced Quoting")
low_execution_probability_idx = data.columns.get_loc("Low Execution Probability")
trades_oppose_quotes_idx = data.columns.get_loc("Trades Oppose Quotes")
cancels_oppose_trades_idx = data.columns.get_loc("Cancels Oppose Trades")
indcs = [ask_price_idx, ask_volume_idx, bid_price_idx, bid_volume_idx, imbalance_idx, frequency_idx, cancellation_rate_idx, high_quoting_activity_idx, unbalanced_quoting_idx, low_execution_probability_idx, trades_oppose_quotes_idx, cancels_oppose_trades_idx]

# Transform the data to numpy
data_numpy = data.to_numpy()

# Initialize and fit the model
model = IsolationForest(contamination=0.01)
# model = IsolationForest(contamination=0.01, n_estimators=1000, max_samples=1.0, max_features=1.0)
model.fit(data_numpy)

# Predict the anomalies in the data
y_pred = model.predict(data_numpy)
y_scores = model.score_samples(data_numpy)
# y_scores_abs = np.abs(y_scores)
y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
anomaly_proba = 1 - y_scores_norm  # The lower the original score, the higher "certainty" it is an anomaly

# Plot the anomalies
for index in indcs:
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Isolation Forest")

    plt.scatter(data_numpy[:, time_idx], data_numpy[:, index], color="green", label="Normal")
    plt.scatter(data_numpy[y_pred == -1, time_idx], data_numpy[y_pred == -1, index], color="red", alpha=anomaly_proba[y_pred == -1], label="Anomaly")

    plt.title(data.columns[index])

    plt.legend()
    plt.grid()

    plt.savefig(f"img/anomaly_detections/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}/IF_{data.columns[index]}.png")
    plt.show()
