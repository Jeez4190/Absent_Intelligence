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
import os

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1. Generar el CSV de Iris (solo si no existe) ===
if not os.path.exists("fisheriris.csv"):
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df.to_csv("fisheriris.csv", index=False)
    print("✅ CSV generado como 'fisheriris.csv'")
else:
    print("📄 El archivo 'fisheriris.csv' ya existe.")

# === 2. Cargar el CSV ===
df = pd.read_csv("fisheriris.csv")
X = df.drop(columns=["species"])
y = df["species"]

# === 3. Dividir en entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 4. Crear y entrenar el modelo ID3 ===
id3 = DecisionTreeClassifier(criterion="entropy", random_state=42)
id3.fit(X_train, y_train)

# === 4.1 Evaluación del modelo ===
y_pred = id3.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 4.2 Matriz de confusión ===
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues",
            xticklabels=id3.classes_, yticklabels=id3.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matriz de Confusión K-NN")
plt.tight_layout()
plt.show()


# === 5. Guardar el modelo ===
joblib.dump(id3, "modelo_id3_iris.joblib")
print("✅ Modelo ID3 guardado como 'modelo_id3_iris.joblib'")

# === 6. Cargar el modelo ===
modelo_cargado = joblib.load("modelo_id3_iris.joblib")

# === 7. Probar el modelo con un nuevo ejemplo (tipo versicolor) ===
nuevo_ejemplo = np.array([[6.0, 2.7, 4.5, 1.5]])  # versicolor
prediccion = modelo_cargado.predict(nuevo_ejemplo)
print(f"🔍 Predicción del nuevo ejemplo: {prediccion[0]}")

# === 8. Evaluar el modelo ===
y_pred = modelo_cargado.predict(X_test)
print("\n🔍 Evaluación del modelo:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))

# === 9. Mostrar el árbol de decisión ===
plt.figure(figsize=(14, 8))
plot_tree(modelo_cargado, 
          feature_names=X.columns, 
          class_names=modelo_cargado.classes_,
          filled=True, rounded=True, fontsize=10)
plt.title("Árbol de Decisión (ID3) para Iris Dataset")
plt.tight_layout()
plt.show()
