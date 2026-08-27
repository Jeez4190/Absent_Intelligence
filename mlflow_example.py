'''
    pip install mlflow scikit-learn pandas
    
    * Despues de ejecutar el análisis, ejecutar el comando para ver los resultados de desempeño:
            mlflow ui
'''

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# =========================================
# 1. Cargar datos
# =========================================

iris = load_iris()

X = iris.data
y = iris.target


# =========================================
# 2. Dividir datos
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================
# 3. Configurar MLflow
# =========================================

mlflow.set_experiment("Iris_KNN")


# =========================================
# 4. Crear experimento
# =========================================

with mlflow.start_run():

    # Hiperparámetros
    n_neighbors = 5
    weights = "distance"

    # Crear modelo
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights
    )

    # Entrenar
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)


    # =========================================
    # 5. Registrar parámetros
    # =========================================

    mlflow.log_param("algorithm", "KNN")
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("weights", weights)


    # =========================================
    # 6. Registrar métrica
    # =========================================

    mlflow.log_metric("accuracy", accuracy)


    # =========================================
    # 7. Registrar modelo
    # =========================================

    mlflow.sklearn.log_model(
        model,
        name="knn_model",
        serialization_format="cloudpickle"
    )


    # =========================================
    # 8. Mostrar resultados
    # =========================================

    print("================================")
    print("      MLflow + KNN + Iris")
    print("================================")

    print(f"Algoritmo: KNN")
    print(f"n_neighbors: {n_neighbors}")
    print(f"weights: {weights}")
    print(f"Accuracy: {accuracy:.4f}")

    print("================================")
    print("Modelo registrado correctamente")
    print("================================")