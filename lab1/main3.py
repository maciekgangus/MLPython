import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

IrisData = sk.datasets.load_iris()
X = IrisData.data
y = IrisData.target

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        X, y, test_size=0.8
    )

neuron = sk.linear_model.Perceptron()
neuron.fit(X_train, y_train)
coefs = neuron.coef_
y_pred = neuron.predict(X_test)
cm = sk.metrics.confusion_matrix(y_test, y_pred)
accuracy = sk.metrics.accuracy_score(y_test, y_pred)
print(coefs)
print(cm)
print(accuracy)


