import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

clf_svm = OneClassSVM(kernel="rbf", degree=3, gamma=0.1, nu=0.01)
y_predict = clf_svm.fit_predict(X)

svm_predict = pd.Series(y_predict).replace([-1,1],[1,0])
svm_anomalies = iris.data[svm_predict == 1]

def plot_OCSVM(i):
    plt.scatter(X[:,i],X[:,i+1],c='green', s=40, edgecolor="k")
    plt.scatter(svm_anomalies[:,i],svm_anomalies[:,i+1],c='red', s=40, edgecolor="k")
    plt.title("OC-SVM Outlier detection between Feature Pair: V{} and V{}".format(i,i+1))
    plt.xlabel("V{}".format(i))
    plt.ylabel("V{}".format(i+1))

plot_OCSVM(0)
plt.show()
