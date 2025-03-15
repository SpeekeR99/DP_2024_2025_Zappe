import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from eval.eval import evaluate

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
data = data.head(10000)

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

# Prepare the data for KFold
kf = KFold(n_splits=5)
y_pred = np.zeros_like(data_numpy[:, 0])
y_scores = np.zeros_like(data_numpy[:, 0])

# Prepare the evaluation results
em_vals = []
mv_vals = []
em_curves = []
mv_curves = []
t = []
axis_alpha = []
amax = -1

# Initialize and fit the model
model = IsolationForest(contamination=0.01)
# model = IsolationForest(contamination=0.01, n_estimators=1000, max_samples=1.0, max_features=1.0)

i = 1
for train_index, test_index in kf.split(data_numpy):
    print(f"KFold {i}")
    i += 1
    # Fit the model
    model.fit(data_numpy[train_index])
    # Predict the anomalies in the data
    y_pred[test_index] = model.predict(data_numpy[test_index])
    y_scores[test_index] = model.decision_function(data_numpy[test_index])
    # Evaluate the model
    em_val, mv_val, em_curve, mv_curve, t_, axis_alpha_, amax_ = evaluate(model, data_numpy[train_index], data_numpy[test_index])
    em_vals.append(em_val)
    mv_vals.append(mv_val)
    em_curves.append(em_curve)
    mv_curves.append(mv_curve)
    t = t_
    axis_alpha = axis_alpha_
    amax = max(amax, amax_)

# Calculate the anomaly probability
y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
anomaly_proba = 1 - y_scores_norm  # The lower the original score, the higher "certainty" it is an anomaly

# Average the evaluation results
em_val = np.mean(em_vals)
mv_val = np.mean(mv_vals)
em_curve = np.mean(em_curves, axis=0)
mv_curve = np.mean(mv_curves, axis=0)

# Print and plot the evaluation results
print(f"EM: {em_val}")
print(f"MV: {mv_val}")
fig = plt.figure(figsize=(20, 10))
fig.suptitle("Isolation Forest")

plt.subplot(121)
plt.plot(t[:amax], em_curve[:amax], lw=1, label=f'Isolation Forest (EM-score = {em_val:.3e})')

plt.ylim([-0.05, 1.05])
plt.xlabel('t')
plt.ylabel('EM(t)')

plt.title('Excess Mass (EM) curves')
plt.legend()

plt.subplot(122)
plt.plot(axis_alpha, mv_curve, lw=1, label=f'Isolation Forest (AUC = {mv_val:.3f})')

plt.xlabel('alpha')
plt.ylabel('MV(alpha)')

plt.title('Mass-Volume (MV) curves')
plt.legend()

plt.savefig(f"img/anomaly_detections/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}/IF_EM_MV.png")
plt.show()

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
