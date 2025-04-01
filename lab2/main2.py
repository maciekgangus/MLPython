import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import time as tm
import seaborn as sns

# Dane
X = sk.datasets.load_digits()['data']
y = sk.datasets.load_digits()['target']
X_norm = X / 16.0

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
    X_norm, y, test_size=0.2, stratify=y
)

# Eksperymenty
experiments = [
    {'arch': (100,), 'activation': 'relu', 'solver': 'adam', 'epochs': 1000},
    {'arch': (100,), 'activation': 'relu', 'solver': 'adam', 'epochs': 1},
    {'arch': (2,), 'activation': 'relu', 'solver': 'adam', 'epochs': 1000},
    {'arch': (100,), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.1, 'epochs': 1000},
    {'arch': (50, 50), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.01, 'epochs': 1000},
    {'arch': (200, 100, 50), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.001, 'epochs': 1000},
    {'arch': (100, 100), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 1.0, 'epochs': 1000},
    {'arch': (100,), 'activation': 'tanh', 'solver': 'adam', 'epochs': 1000},
    {'arch': (300,), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.05, 'epochs': 1000},
]

# Przygotowanie rysunku
num_experiments = len(experiments)
cols = 3
rows = (num_experiments + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
axes = axes.flatten()

results = []

# Główna pętla
for i, (exp, ax) in enumerate(zip(experiments, axes), 1):
    model = sk.neural_network.MLPClassifier(
        hidden_layer_sizes=exp['arch'],
        activation=exp['activation'],
        solver=exp['solver'],
        learning_rate_init=exp.get('learning_rate_init', 0.001),
        max_iter=exp['epochs'],
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

    # Rysowanie macierzy pomyłek
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax, cbar=False)
    ax.set_title(
        f"Exp {i}: {exp['arch']}, akt={exp['activation']}, sol={exp['solver']}, "
        f"lr={exp.get('learning_rate_init', 'default')}, ep={exp['epochs']}\n"
        f"Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1:.2f}, Time={training_time}s",
        fontsize=9
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    results.append({
        'Exp': i,
        'Arch': str(exp['arch']),
        'Akt': exp['activation'],
        'Sol': exp['solver'],
        'LR': exp.get('learning_rate_init', 'default'),
        'Epochs': exp['epochs'],
        'Train Acc': sk.metrics.accuracy_score(y_train, y_train_pred),
        'Test Acc': acc,
        'Test Precision': prec,
        'Test Recall': rec,
        'Test F1': f1,
        'Training Time': training_time,
    })

# Usuwanie pustych wykresów
for i in range(len(experiments), len(axes)):
    fig.delaxes(axes[i])

fig.tight_layout()
plt.savefig("macierze_pomylek.png", dpi=300)
plt.show()

# Zapis wyników
df_results = pd.DataFrame(results)
metrics = ['Train Acc', 'Test Acc', 'Test Precision', 'Test Recall', 'Test F1']
df_results[metrics] = df_results[metrics].round(4)
df_results.to_csv("wyniki_mlp.csv", index=False)
