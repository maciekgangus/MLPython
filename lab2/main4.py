from ucimlrepo import fetch_ucirepo
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import time as tm
import seaborn as sns

# fetch dataset
yeast = fetch_ucirepo(id=110)

# data (as pandas dataframes)
X = yeast.data.features
y = yeast.data.targets


X = X.to_numpy()
y = y.to_numpy()

outlier_counts = np.ones(X.shape[0], dtype=int)


# Outliery
for i in range(X.shape[1]):
    if i in [5, 6]:
        break
    Q1 = np.percentile(X[:, i], 25)
    Q3 = np.percentile(X[:, i], 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
    outlier_counts += outlier_mask

mask = outlier_counts < 2
Xf = X[mask]
yf = y[mask]
yf = yf.ravel()

print(f"Oryginalny rozmiar: {X.shape}")
print(f"Po usunięciu outlierów: {Xf.shape}")
print("MinMax przed: ", Xf.min(axis=0), Xf.max(axis=0))


#MinMax
X_min = Xf.min(axis=0)
X_max = Xf.max(axis=0)
range = X_max - X_min
range[range == 0] = 1
X_minmax_final = (Xf - X_min) / range
print("minmax po: ", X_min, X_max)

experiments = [
    {
        'arch': (100, 100),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.001,
        'max_iter': 2000
    },
    {
        'arch': (300, 200, 100),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.0005,
        'max_iter': 3000
    },
    {
        'arch': (50, 50),
        'activation': 'tanh',
        'solver': 'adam',
        'learning_rate_init': 0.001,
        'max_iter': 2000
    },
    {
        'arch': (100,),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.005,
        'max_iter': 1500
    },
    {
        'arch': (150, 75),
        'activation': 'tanh',
        'solver': 'sgd',
        'learning_rate_init': 0.01,
        'max_iter': 3000
    },
    {
        'arch': (100, 100, 100),
        'activation': 'relu',
        'solver': 'sgd',
        'learning_rate_init': 0.05,
        'max_iter': 3000
    },
    {
        'arch': (200,),
        'activation': 'logistic',
        'solver': 'adam',
        'learning_rate_init': 0.001,
        'max_iter': 2000
    },
]

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
    X_minmax_final, yf, test_size=0.2, stratify=yf
)

results = []

# Przygotowanie wykresów
cols = 3
rows = (len(experiments) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
axes = axes.flatten()

for i, (exp, ax) in enumerate(zip(experiments, axes)):
    model = sk.neural_network.MLPClassifier(
        hidden_layer_sizes=exp['arch'],
        activation=exp['activation'],
        solver=exp['solver'],
        learning_rate_init=exp.get('learning_rate_init'),
        max_iter=exp['max_iter'],
        random_state=42,
        early_stopping=False
    )

    start = tm.time()
    model.fit(X_train, y_train)
    end = tm.time()
    training_time = round(end - start, 4)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    acc = sk.metrics.accuracy_score(y_test, y_test_pred)
    prec = sk.metrics.precision_score(y_test, y_test_pred, average='weighted')
    rec = sk.metrics.recall_score(y_test, y_test_pred, average='weighted')
    f1 = sk.metrics.f1_score(y_test, y_test_pred, average='weighted')
    cm = sk.metrics.confusion_matrix(y_test, y_test_pred)

    # Rysowanie macierzy
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax, cbar=False)
    ax.set_title(
        f"Exp {i}: {exp['arch']}, akt={exp['activation']}, sol={exp['solver']}, "
        f"lr={exp.get('learning_rate_init', 'default')}, ep={exp['max_iter']}\n"
        f"Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1:.2f}, Time={training_time}s",
        fontsize=9
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    results.append({
        'Eksperyment': i,
        'Architektura': str(exp['arch']),
        'Aktywacja': exp['activation'],
        'Solver': exp['solver'],
        'LR': exp.get('learning_rate_init', 'default'),
        'Train Acc': sk.metrics.accuracy_score(y_train, y_train_pred),
        'Test Acc': acc,
        'Test Precision': prec,
        'Test Recall': rec,
        'Test F1': f1,
        'Training Time': training_time,
    })

# Usunięcie pustych wykresów
for j in range(len(experiments), len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.savefig("macierze_pomylek_yeast.png", dpi=300)
plt.show()



