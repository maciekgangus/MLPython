import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import time as tm

def_path = os.getcwd()

DataFrame = pd.read_csv(f"{def_path}/data/medicine.txt", delimiter=',')
data_array = DataFrame.to_numpy()
print(data_array)

Q1_1 = np.percentile(data_array[:, 0], 25)
Q3_1 = np.percentile(data_array[:, 0], 75)
IQR_1 = Q3_1 - Q1_1

lower_bound_1 = Q1_1 - 1.5 * IQR_1
upper_bound_1 = Q3_1 + 1.5 * IQR_1

data_array_filtered = data_array[(data_array[:, 0] > lower_bound_1) & (data_array[:, 0] < upper_bound_1)]

Q1_2 = np.percentile(data_array_filtered[:, 1], 25)
Q3_2 = np.percentile(data_array_filtered[:, 1], 75)
IQR_2 = Q3_2 - Q1_2

lower_bound_2 = Q1_2 - 1.5 * IQR_2
upper_bound_2 = Q3_2 + 1.5 * IQR_2

data_array_final = data_array_filtered[
    (data_array_filtered[:, 1] > lower_bound_2) & (data_array_filtered[:, 1] < upper_bound_2)]

print("Oryginalny rozmiar:", data_array.shape)
print("Po usunięciu odstających w pierwszej kolumnie:", data_array_filtered.shape)
print("Po usunięciu odstających w drugiej kolumnie:", data_array_final.shape)

yf = data_array_final[:, 2]
Xf = data_array_final[:, :-1]
print(Xf[:, 0].max())

X_min = Xf.min(axis=0)
X_max = Xf.max(axis=0)
X_minmax_final = (Xf - X_min) / (X_max - X_min)

print("minmax po: ", X_min, X_max)

Xf_train, Xf_test, yf_train, yf_test = sk.model_selection.train_test_split(
    X_minmax_final, yf, test_size=0.2, stratify=yf
)

def plot_decision_boundary(model, X, y, ax):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

architectures = [
    (3,),
    (50,),
    (2, 2),
    (100,),
    (1000, 1000),
    (300, 200, 100),
    (50, 50, 50),
    (2, 2, 2, 2, 2, 2, 2, 2, 2),
]

fig, axes = plt.subplots(len(architectures), 2, figsize=(14, 36))
axes = np.array(axes)

for i, arch in enumerate(architectures):
    print(f"\n=== Sieć {i + 1}: Architektura {arch} ===")

    model = sk.neural_network.MLPClassifier(hidden_layer_sizes=arch, max_iter=2000, random_state=42)
    start = tm.time()
    model.fit(Xf_train, yf_train)
    end = tm.time()

    # TESTOWE
    y_test_pred = model.predict(Xf_test)
    acc_test = sk.metrics.accuracy_score(yf_test, y_test_pred)
    f1_test = sk.metrics.f1_score(yf_test, y_test_pred, average='weighted')
    precision_test = sk.metrics.precision_score(yf_test, y_test_pred, average='weighted')  # Precyzja
    recall_test = sk.metrics.recall_score(yf_test, y_test_pred, average='weighted')
    plot_decision_boundary(model, Xf_test, yf_test, axes[i, 1])
    axes[i, 1].set_title(f"Test: {arch}\nAcc: {acc_test:.2f}, F1: {f1_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, Time: {end-start:.2f}s")

    # TRENINGOWE
    y_train_pred = model.predict(Xf_train)
    acc_test = sk.metrics.accuracy_score(yf_train, y_train_pred)
    f1_test = sk.metrics.f1_score(yf_train, y_train_pred, average='weighted')
    precision_test = sk.metrics.precision_score(yf_train, y_train_pred, average='weighted')  # Precyzja
    recall_test = sk.metrics.recall_score(yf_train, y_train_pred, average='weighted')
    plot_decision_boundary(model, Xf_train, yf_train, axes[i, 0])
    axes[i, 0].set_title(f"Train: {arch}\nAcc: {acc_test:.2f}, F1: {f1_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, Time: {end-start:.2f}s")

plt.tight_layout()
plt.show()

