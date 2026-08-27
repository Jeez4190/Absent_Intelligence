'''
    pip install mlflow scikit-learn pandas
    
    * Despues de ejecutar el análisis, ejecutar el comando para ver los resultados de desempeño:
            mlflow ui
    
'''

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =====================================================
# 1. Cargar dataset
# =====================================================

iris = load_iris()

X = iris.data
y = iris.target


# =====================================================
# 2. Dividir dataset
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 3. Configurar MLflow
# =====================================================

mlflow.set_experiment("Iris_Classification")


# =====================================================
# 4. Definir modelos
# =====================================================

models = {

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "SVM": SVC(
        C=1.0,
        kernel="rbf"
    ),

    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=200
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}


# =====================================================
# 5. Entrenar modelos
# =====================================================

for name, model in models.items():

    print("\n===================================")
    print(f"Entrenando: {name}")
    print("===================================")

    with mlflow.start_run(run_name=name):

        # ---------------------------------------------
        # Entrenamiento
        # ---------------------------------------------

        model.fit(X_train, y_train)

        # ---------------------------------------------
        # Predicción
        # ---------------------------------------------

        y_pred = model.predict(X_test)

        # ---------------------------------------------
        # Evaluación
        # ---------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")

        # ---------------------------------------------
        # Registrar tipo de modelo
        # ---------------------------------------------

        mlflow.log_param(
            "model_type",
            name
        )

        # ---------------------------------------------
        # Registrar hiperparámetros
        # ---------------------------------------------

        for parameter, value in model.get_params().items():

            if isinstance(value, (str, int, float, bool)):

                mlflow.log_param(
                    parameter,
                    value
                )

        # ---------------------------------------------
        # Registrar métrica
        # ---------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # ---------------------------------------------
        # Registrar modelo
        # ---------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )


print("\n===================================")
print("Todos los modelos fueron registrados")
print("===================================")