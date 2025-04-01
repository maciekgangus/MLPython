import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

def_path = os.getcwd()


# ---------
DataFrame = pd.read_csv(f"{def_path}/data/fuel.txt", delimiter=',')
data_array = DataFrame.to_numpy()

Xf = data_array[:, :-1]
yf = np.array([0 if i == "A" else 1 for i in data_array[:, -1]])
X_min = Xf.min(axis=0)
X_max = Xf.max(axis=0)
X_minmax_s = (Xf- X_min) / (X_max - X_min)

X_mean = Xf.mean(axis=0)
X_std = Xf.std(axis=0)
X_std_s = (Xf - X_mean) / X_std



accuracies = []

for i in range(5):
    indices = np.random.permutation(len(Xf))
    Xf_shuffled, yf_shuffled = Xf[indices], yf[indices]

    Xf_train, Xf_test, yf_train, yf_test = sk.model_selection.train_test_split(
        Xf_shuffled, yf_shuffled, test_size=0.5, random_state=2
    )

    neuronf = sk.linear_model.Perceptron()
    neuronf.fit(Xf_train, yf_train)
    coef = neuronf.coef_
    print(coef)

    yf_pred = neuronf.predict(Xf_test)
    cmf = sk.metrics.confusion_matrix(yf_test, yf_pred)
    print(cmf)
    accuracyf = sk.metrics.accuracy_score(yf_test, yf_pred)
    accuracies.append(accuracyf)

print(accuracies)