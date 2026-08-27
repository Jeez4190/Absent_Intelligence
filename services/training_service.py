import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# CONFIGURACIÓN
# ==========================================

MODEL_FOLDER = "models"


# ==========================================
# ENTRENAR MODELO
# ==========================================

def entrenar_modelo(
    X,
    y,
    algoritmo,
    test_size=0.30,
    parametros=None
):
    """
    Entrena un modelo de Machine Learning.

    Parámetros
    ----------
    X : pandas.DataFrame
        Características del dataset.

    y : pandas.Series
        Variable objetivo.

    algoritmo : str
        Algoritmo a utilizar:
        - id3
        - knn

    test_size : float
        Porcentaje de datos utilizado para prueba.
        Ejemplo:
        0.30 -> 70% entrenamiento / 30% prueba
        0.10 -> 90% entrenamiento / 10% prueba

    parametros : dict
        Parámetros configurables del algoritmo.

    Retorna
    -------
    modelo : modelo sklearn

    resultados : dict
        Métricas y resultados del entrenamiento.
    """


    # ==========================================
    # VALIDACIONES GENERALES
    # ==========================================

    if parametros is None:
        parametros = {}


    if X is None or y is None:
        raise ValueError(
            "Los datos de entrenamiento no pueden estar vacíos."
        )


    if len(X) == 0:
        raise ValueError(
            "El dataset no contiene datos."
        )


    if len(X) != len(y):
        raise ValueError(
            "X e y deben tener la misma cantidad de registros."
        )


    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            "El porcentaje de prueba debe estar entre 0 y 1."
        )


    # ==========================================
    # DIVISIÓN TRAIN / TEST
    # ==========================================

    try:

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=test_size,

            random_state=42,

            stratify=y

        )

    except ValueError as e:

        raise ValueError(
            f"No fue posible dividir el dataset: {e}"
        )


    # ==========================================
    # CREAR MODELO ID3
    # ==========================================

    if algoritmo == "id3":

        max_depth = parametros.get(
            "max_depth",
            ""
        )


        # --------------------------------------
        # Convertir profundidad
        # --------------------------------------

        if max_depth in ("", None):

            max_depth = None

        else:

            try:

                max_depth = int(max_depth)

            except ValueError:

                raise ValueError(
                    "La profundidad máxima de ID3 "
                    "debe ser un número entero."
                )


            if max_depth < 1:

                raise ValueError(
                    "La profundidad máxima debe ser "
                    "mayor o igual a 1."
                )


        # --------------------------------------
        # Crear ID3
        # --------------------------------------

        modelo = DecisionTreeClassifier(

            criterion="entropy",

            max_depth=max_depth,

            random_state=42

        )


    # ==========================================
    # CREAR MODELO K-NN
    # ==========================================

    elif algoritmo == "knn":

        # --------------------------------------
        # Número de vecinos
        # --------------------------------------

        try:

            n_neighbors = int(
                parametros.get(
                    "n_neighbors",
                    5
                )
            )

        except ValueError:

            raise ValueError(
                "El número de vecinos K "
                "debe ser un número entero."
            )


        if n_neighbors < 1:

            raise ValueError(
                "El número de vecinos K "
                "debe ser mayor o igual a 1."
            )


        if n_neighbors > len(X_train):

            raise ValueError(
                "K no puede ser mayor que la cantidad "
                "de datos de entrenamiento."
            )


        # --------------------------------------
        # Pesos
        # --------------------------------------

        weights = parametros.get(
            "weights",
            "uniform"
        )


        if weights not in [
            "uniform",
            "distance"
        ]:

            raise ValueError(
                "Los pesos deben ser "
                "'uniform' o 'distance'."
            )


        # --------------------------------------
        # Métrica
        # --------------------------------------

        metric = parametros.get(
            "metric",
            "minkowski"
        )


        metricas_validas = [

            "minkowski",

            "euclidean",

            "manhattan"

        ]


        if metric not in metricas_validas:

            raise ValueError(
                "Métrica no válida. "
                "Utiliza minkowski, euclidean "
                "o manhattan."
            )


        # --------------------------------------
        # Crear K-NN
        # --------------------------------------

        modelo = KNeighborsClassifier(

            n_neighbors=n_neighbors,

            weights=weights,

            metric=metric

        )


    # ==========================================
    # ALGORITMO NO EXISTE
    # ==========================================

    else:

        raise ValueError(
            f"Algoritmo no soportado: {algoritmo}"
        )


    # ==========================================
    # ENTRENAMIENTO
    # ==========================================

    modelo.fit(

        X_train,

        y_train

    )


    # ==========================================
    # PREDICCIONES
    # ==========================================

    y_pred = modelo.predict(

        X_test

    )


    # ==========================================
    # MÉTRICAS
    # ==========================================

    accuracy = accuracy_score(

        y_test,

        y_pred

    )


    precision = precision_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    # ==========================================
    # MATRIZ DE CONFUSIÓN
    # ==========================================

    matriz = confusion_matrix(

        y_test,

        y_pred

    )


    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================

    reporte = classification_report(

        y_test,

        y_pred,

        zero_division=0

    )


    # ==========================================
    # RESULTADOS
    # ==========================================

    resultados = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "confusion_matrix": matriz.tolist(),

    "classification_report": reporte,

    "train_size": len(X_train),

    "test_size": len(X_test),

    "test_percentage": test_size,

    "algorithm": algoritmo,

    "parameters": parametros,

    "X_test": X_test,

    "y_test": y_test

}


    # ==========================================
    # RETORNAR
    # ==========================================

    return modelo, resultados


# ==========================================
# GUARDAR MODELO
# ==========================================

def guardar_modelo(
    modelo,
    nombre_modelo
):
    """
    Guarda un modelo entrenado en formato .joblib.
    """


    # Crear carpeta models si no existe

    os.makedirs(

        MODEL_FOLDER,

        exist_ok=True

    )


    # Asegurar extensión

    if not nombre_modelo.endswith(
        ".joblib"
    ):

        nombre_modelo += ".joblib"


    # Ruta completa

    ruta = os.path.join(

        MODEL_FOLDER,

        nombre_modelo

    )


    # Guardar

    joblib.dump(

        modelo,

        ruta

    )


    return ruta


# ==========================================
# CARGAR MODELO
# ==========================================

def cargar_modelo(
    nombre_modelo
):
    """
    Carga un modelo .joblib previamente guardado.
    """


    if not nombre_modelo.endswith(
        ".joblib"
    ):

        nombre_modelo += ".joblib"


    ruta = os.path.join(

        MODEL_FOLDER,

        nombre_modelo

    )


    if not os.path.exists(ruta):

        raise FileNotFoundError(

            f"No se encontró el modelo: "
            f"{nombre_modelo}"

        )


    modelo = joblib.load(

        ruta

    )


    return modelo