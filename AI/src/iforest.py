import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the data
filepath = "../data/lobster_best_prices_only.csv"
data = pd.read_csv(filepath)
data_numpy = data.to_numpy()

# initialize and fit the model
clf = IsolationForest(contamination=0.01, n_estimators=1000, max_samples=1.0, max_features=1.0)
clf.fit(data_numpy)

# predict the anomalies in the data
y_pred_train = clf.predict(data_numpy)

# plot the anomalies
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

axes[0][0].scatter(data_numpy[y_pred_train==1, 0], data_numpy[y_pred_train==1, 1], color="green", label="Normal Ask")
axes[0][0].scatter(data_numpy[y_pred_train==-1, 0], data_numpy[y_pred_train==-1, 1], color="red", label="Anomaly Ask")
axes[0][0].set_title("Ask Price")
axes[0][0].legend()
axes[0][0].grid()

axes[0][1].scatter(data_numpy[y_pred_train==1, 0], data_numpy[y_pred_train==1, 3], color="green", label="Normal Bid")
axes[0][1].scatter(data_numpy[y_pred_train==-1, 0], data_numpy[y_pred_train==-1, 3], color="red", label="Anomaly Bid")
axes[0][1].set_title("Bid Price")
axes[0][1].legend()
axes[0][1].grid()

axes[1][0].scatter(data_numpy[y_pred_train==1, 0], data_numpy[y_pred_train==1, 5], color="green", label="Normal Imbalance Index")
axes[1][0].scatter(data_numpy[y_pred_train==-1, 0], data_numpy[y_pred_train==-1, 5], color="red", label="Anomaly Imbalance Index")
axes[1][0].set_title("Imbalance Index")
axes[1][0].legend()
axes[1][0].grid()

axes[1][1].scatter(data_numpy[y_pred_train==1, 0], data_numpy[y_pred_train==1, 6], color="green", label="Normal Frequency")
axes[1][1].scatter(data_numpy[y_pred_train==-1, 0], data_numpy[y_pred_train==-1, 6], color="red", label="Anomaly Frequency")
axes[1][1].set_title("Frequency of incoming messages")
axes[1][1].legend()
axes[1][1].grid()

plt.show()
