#!/usr/bin/env python3

'''
    Iniciar:
        source venv/bin/activate

    Instalar:
        pip install pandas scikit-learn matplotlib seaborn joblib

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1. Generar y guardar el CSV de Iris desde sklearn ===
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.to_csv("fisheriris.csv", index=False)
print("✅ CSV generado como 'fisheriris.csv'")

# === 2. Cargar datos del CSV ===
df = pd.read_csv("fisheriris.csv")
X = df.drop(columns=["species"])
y = df["species"]

# === 3. Dividir conjunto de datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 4. Crear y entrenar el modelo ===
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# === 4.1 Evaluación del modelo ===
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 7. Matriz de confusión ===
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues",
            xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matriz de Confusión K-NN")
plt.tight_layout()
plt.show()


# === 5. Guardar el modelo entrenado ===
joblib.dump(knn, "modelo_knn_iris.joblib")
print("✅ Modelo guardado como 'modelo_knn_iris.joblib'")

# === 6. Cargar el modelo guardado ===
modelo_cargado = joblib.load("modelo_knn_iris.joblib")

# === 7. Probar con un nuevo ejemplo (tipo versicolor) ===
nuevo_ejemplo = np.array([[6.0, 2.7, 4.5, 1.5]])
prediccion = modelo_cargado.predict(nuevo_ejemplo)
print(f"🔍 Predicción del nuevo ejemplo: {prediccion[0]}")

# === 8. Evaluar el modelo ===
y_pred = modelo_cargado.predict(X_test)
print("\n🔍 Evaluación del modelo:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))

# === 9. Graficar los datos y vecinos más cercanos ===

# Usamos solo 2 características para visualizar (petal length vs petal width)
feature_x = "petal length (cm)"
feature_y = "petal width (cm)"

# Convertir a numpy
X_vis = df[[feature_x, feature_y]].values
y_vis = df["species"].values

# Reentrenar un modelo solo con esas dos características
knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_vis, y_vis)

# Obtener los vecinos del nuevo ejemplo
nuevo_vis = np.array([[4.5, 1.5]])  # mismo ejemplo, pero solo 2 características
distancias, indices = knn_vis.kneighbors(nuevo_vis)

# === Visualización ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=feature_x, y=feature_y, hue="species", data=df, palette="Set2", s=60)

# Marcar el nuevo ejemplo
plt.scatter(nuevo_vis[0][0], nuevo_vis[0][1], color='red', label='Nuevo ejemplo', s=100, edgecolor='black', marker='X')

# Dibujar líneas a los vecinos más cercanos
for idx in indices[0]:
    vecino = X_vis[idx]
    plt.plot([nuevo_vis[0][0], vecino[0]], [nuevo_vis[0][1], vecino[1]], 'k--', linewidth=0.8)

plt.title("K-NN en espacio 2D (petal length vs petal width)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
