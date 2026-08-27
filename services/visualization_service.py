import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# CONFIGURACIÓN
# ============================================================

GRAPH_FOLDER = os.path.join(
    "static",
    "graphs"
)


# ============================================================
# CREAR CARPETA
# ============================================================

def crear_carpeta_graficas():

    os.makedirs(
        GRAPH_FOLDER,
        exist_ok=True
    )


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def generar_matriz_confusion(
    modelo,
    X_test,
    y_test,
    nombre_archivo="confusion_matrix.png"
):

    crear_carpeta_graficas()

    y_pred = modelo.predict(
        X_test
    )

    matriz = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=modelo.classes_,
        yticklabels=modelo.classes_
    )

    plt.xlabel(
        "Predicción"
    )

    plt.ylabel(
        "Valor real"
    )

    plt.title(
        "Matriz de Confusión"
    )

    plt.tight_layout()

    ruta = os.path.join(
        GRAPH_FOLDER,
        nombre_archivo
    )

    plt.savefig(
        ruta,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return ruta


# ============================================================
# ÁRBOL ID3
# ============================================================

def generar_arbol_id3(
    modelo,
    X,
    nombre_archivo="arbol_id3.png"
):

    crear_carpeta_graficas()

    plt.figure(
        figsize=(16, 9)
    )

    plot_tree(
        modelo,

        feature_names=X.columns,

        class_names=modelo.classes_,

        filled=True,

        rounded=True,

        fontsize=9
    )

    plt.title(
        "Árbol de Decisión ID3"
    )

    plt.tight_layout()

    ruta = os.path.join(
        GRAPH_FOLDER,
        nombre_archivo
    )

    plt.savefig(
        ruta,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return ruta


# ============================================================
# GRÁFICA K-NN
# ============================================================

def generar_grafica_knn(
    df,
    columnas,
    objetivo,
    k,
    nombre_archivo="knn_2d.png"
):

    crear_carpeta_graficas()


    # ========================================================
    # VALIDAR DOS CARACTERÍSTICAS
    # ========================================================

    if len(columnas) != 2:

        raise ValueError(
            "Para generar la gráfica de K-NN "
            "debes seleccionar exactamente "
            "2 atributos."
        )


    feature_x = columnas[0]

    feature_y = columnas[1]


    # ========================================================
    # DATOS
    # ========================================================

    X = df[
        columnas
    ].values

    y = df[
        objetivo
    ].values


    # ========================================================
    # ENTRENAR KNN PARA VISUALIZACIÓN
    # ========================================================

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X,
        y
    )


    # ========================================================
    # EJEMPLO
    # ========================================================

    nuevo_ejemplo = X.mean(
        axis=0
    ).reshape(
        1,
        -1
    )


    # ========================================================
    # VECINOS
    # ========================================================

    distancias, indices = (
        knn.kneighbors(
            nuevo_ejemplo
        )
    )


    # ========================================================
    # FIGURA
    # ========================================================

    plt.figure(
        figsize=(9, 7)
    )


    # ========================================================
    # DATASET
    # ========================================================

    sns.scatterplot(

        x=feature_x,

        y=feature_y,

        hue=objetivo,

        data=df,

        s=70

    )


    # ========================================================
    # NUEVO EJEMPLO
    # ========================================================

    plt.scatter(

        nuevo_ejemplo[0][0],

        nuevo_ejemplo[0][1],

        color="red",

        label="Nuevo ejemplo",

        s=150,

        edgecolor="black",

        marker="X"

    )


    # ========================================================
    # LINEAS A LOS VECINOS
    # ========================================================

    for idx in indices[0]:

        vecino = X[idx]


        plt.plot(

            [
                nuevo_ejemplo[0][0],
                vecino[0]
            ],

            [
                nuevo_ejemplo[0][1],
                vecino[1]
            ],

            "k--",

            linewidth=0.8

        )


    # ========================================================
    # TITULO
    # ========================================================

    plt.title(

        f"K-NN - {k} vecinos más cercanos"

    )


    plt.xlabel(
        feature_x
    )


    plt.ylabel(
        feature_y
    )


    plt.legend()


    plt.grid(
        True
    )


    plt.tight_layout()


    # ========================================================
    # GUARDAR
    # ========================================================

    ruta = os.path.join(

        GRAPH_FOLDER,

        nombre_archivo

    )


    plt.savefig(

        ruta,

        dpi=150,

        bbox_inches="tight"

    )


    plt.close()


    return ruta