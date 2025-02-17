import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_svm = OneClassSVM(kernel="rbf", degree=3, gamma=0.1, nu=0.01)
y_predict_train = clf_svm.fit_predict(X_train)
y_predict_test = clf_svm.predict(X_test)

svm_predict_train = pd.Series(y_predict_train).replace([-1,1],[1,0])
svm_anomalies_train = X_train[svm_predict_train == 1]
svm_predict_test = pd.Series(y_predict_test).replace([-1,1],[1,0])
svm_anomalies_test = X_test[svm_predict_test == 1]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(X_train[:, 0], X_train[:, 1], color='green', label='Normal')
axes[0].scatter(svm_anomalies_train[:, 0], svm_anomalies_train[:, 1], color='red', label='Anomaly')
axes[0].set_title('Training Data')
axes[0].legend()

axes[1].scatter(X_test[:, 0], X_test[:, 1], color='green', label='Normal')
axes[1].scatter(svm_anomalies_test[:, 0], svm_anomalies_test[:, 1], color='red', label='Anomaly')
axes[1].set_title('Test Data')
axes[1].legend()

plt.tight_layout()
plt.show()
