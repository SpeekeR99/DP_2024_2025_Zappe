import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the data
DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"
filepath = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster_augmented.csv"
data = pd.read_csv(filepath)
# data = data[["Time", "Ask Price 1", "Bid Price 1", "Imbalance Index", "Frequency of Incoming Messages", "Cancellations Rate"]]
data = data.head(100000)

# Get the indices of the columns
try:
    time_idx = data.columns.get_loc("Time")
    ask_price_idx = data.columns.get_loc("Ask Price 1")
    bid_price_idx = data.columns.get_loc("Bid Price 1")
    imbalance_idx = data.columns.get_loc("Imbalance Index")
    frequency_idx = data.columns.get_loc("Frequency of Incoming Messages")
    cancellation_rate_idx = data.columns.get_loc("Cancellations Rate")
except KeyError as e:
    print(f"The input file does not contain the necessary column {e}.")
    exit(1)

# Transform the data to numpy
data_numpy = data.to_numpy()
mid_price = (data_numpy[:, ask_price_idx] + data_numpy[:, bid_price_idx]) / 2

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
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle("Isolation Forest")

axes[0][0].scatter(data_numpy[:, time_idx], mid_price, color="green", label="Normal Mid Price")
axes[0][0].scatter(data_numpy[y_pred == -1, time_idx], mid_price[y_pred == -1], color="red", alpha=anomaly_proba[y_pred == -1], label="Anomaly Mid Price")
axes[0][0].set_title("Mid Price")
axes[0][0].legend()
axes[0][0].grid()

axes[0][1].scatter(data_numpy[:, time_idx], data_numpy[:, imbalance_idx], color="green", label="Normal Imbalance Index")
axes[0][1].scatter(data_numpy[y_pred == -1, time_idx], data_numpy[y_pred == -1, imbalance_idx], color="red", alpha=anomaly_proba[y_pred == -1], label="Anomaly Imbalance Index")
axes[0][1].set_title("Imbalance Index")
axes[0][1].legend()
axes[0][1].grid()

axes[1][0].scatter(data_numpy[:, time_idx], data_numpy[:, frequency_idx], color="green", label="Normal Frequency")
axes[1][0].scatter(data_numpy[y_pred == -1, time_idx], data_numpy[y_pred == -1, frequency_idx], color="red", alpha=anomaly_proba[y_pred == -1], label="Anomaly Frequency")
axes[1][0].set_title("Frequency of incoming messages")
axes[1][0].legend()
axes[1][0].grid()

axes[1][1].scatter(data_numpy[:, time_idx], data_numpy[:, cancellation_rate_idx], color="green", label="Normal Cancellations Rate")
axes[1][1].scatter(data_numpy[y_pred == -1, time_idx], data_numpy[y_pred == -1, cancellation_rate_idx], color="red", alpha=anomaly_proba[y_pred == -1], label="Anomaly Cancellations Rate")
axes[1][1].set_title("Cancellations Rate")
axes[1][1].legend()
axes[1][1].grid()

plt.show()
