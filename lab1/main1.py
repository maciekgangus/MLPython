import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

# Generowanie danych
K1 = np.random.normal(loc=[0, -1], scale=1, size=(100, 2))
K2 = np.random.normal(loc=[1, 1], scale=1, size=(100, 2))

X = np.concatenate((K1, K2))
y = [0 if i < 100 else 1 for i in range(200)]

# Tworzenie siatki wykresów (2 kolumny, 2 wiersze)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # Zwiększony obszar

# Lista rozmiarów zbioru treningowego
sizes = [5, 10, 20, 100]

for i, size in enumerate(sizes):
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        X, y, test_size=(200 - size) / 200, random_state=2
    )

    # Trenowanie modelu perceptronu
    neuron = sk.linear_model.Perceptron()
    neuron.fit(X_train, y_train)

    # Wyznaczanie prostej decyzyjnej
    x1 = np.linspace(-3, 3, 1000)
    x2 = -(1.0 / neuron.coef_[0][1]) * (neuron.coef_[0][0] * x1 + neuron.intercept_[0])

    # Pobranie odpowiedniego podwykresu (2x2)
    ax = axes[i // 2, i % 2]

    # Rysowanie punktów
    ax.scatter(K1[:, 0], K1[:, 1], label="k1")
    ax.scatter(K2[:, 0], K2[:, 1], label="k2")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Rozmiar treningowy={size} \n {round(neuron.coef_[0][0], 6)}x1 + {round(neuron.coef_[0][1], 6)}x2 + {neuron.intercept_[0]}")
    # Predykcja na zbiorze testowym
    y_pred = neuron.predict(X_test)

    # Obliczanie macierzy konfuzji i dokładności
    cm = sk.metrics.confusion_matrix(y_test, y_pred)
    accuracy = sk.metrics.accuracy_score(y_test, y_pred)
    print(cm)
    print(round(accuracy, 2))


    # Rysowanie linii decyzyjnej
    ax.plot(x1, x2, '-r')



# Dopasowanie układu
plt.tight_layout()
plt.show()






