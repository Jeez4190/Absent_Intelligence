import os
import pandas as pd


DATASET_FOLDER = "datasets"


def obtener_datasets():
    """
    Obtiene todos los archivos CSV disponibles
    en la carpeta datasets.
    """

    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

    archivos = []

    for archivo in os.listdir(DATASET_FOLDER):

        if archivo.lower().endswith(".csv"):
            archivos.append(archivo)

    return archivos


def cargar_dataset(nombre_archivo):
    """
    Carga un archivo CSV.
    """

    ruta = os.path.join(
        DATASET_FOLDER,
        nombre_archivo
    )

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el dataset: {nombre_archivo}"
        )

    return pd.read_csv(ruta)


def obtener_informacion(df):
    """
    Obtiene información general del dataset.
    """

    return {
        "filas": df.shape[0],

        "columnas": df.shape[1],

        "nombres_columnas": df.columns.tolist(),

        "columnas_numericas": df.select_dtypes(
            include="number"
        ).columns.tolist(),

        "columnas_categoricas": df.select_dtypes(
            exclude="number"
        ).columns.tolist(),

        "valores_nulos": int(
            df.isnull().sum().sum()
        )
    }


def obtener_columnas(df):
    """
    Obtiene los nombres de las columnas.
    """

    return df.columns.tolist()


def preparar_datos(df, columna_objetivo):
    """
    Separa las características X de la variable objetivo y.
    """

    if columna_objetivo not in df.columns:
        raise ValueError(
            f"La columna '{columna_objetivo}' no existe."
        )

    X = df.drop(
        columns=[columna_objetivo]
    )

    y = df[columna_objetivo]

    return X, y


def obtener_preview(df, cantidad=5):
    """
    Obtiene las primeras filas del dataset.
    """

    return df.head(cantidad)