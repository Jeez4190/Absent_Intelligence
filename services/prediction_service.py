import json
import os

import numpy as np
import pandas as pd
from sklearn.tree import _tree

from services.gemini_service import explicar_como_experto
from services.training_service import MODEL_FOLDER, cargar_modelo


DESCRIPCIONES_PLANTAS = {
    "setosa": (
        "Iris setosa suele tener pétalos cortos y estrechos. "
        "Es la especie más fácil de separar del resto."
    ),
    "versicolor": (
        "Iris versicolor tiene pétalos de tamaño intermedio: "
        "más grandes que setosa y más pequeños que virginica."
    ),
    "virginica": (
        "Iris virginica suele tener pétalos más largos y anchos. "
        "Es la especie de mayor tamaño en este conjunto."
    ),
}


def guardar_metadatos_modelo(nombre_modelo, metadatos):
    ruta = _ruta_metadatos(nombre_modelo)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    return ruta


def cargar_metadatos_modelo(nombre_modelo):
    ruta = _ruta_metadatos(nombre_modelo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No se encontraron los metadatos del modelo. "
            "Vuelve a entrenar el modelo para poder probarlo."
        )
    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def predecir_muestra(nombre_modelo, valores, df):
    metadatos = cargar_metadatos_modelo(nombre_modelo)
    modelo = cargar_modelo(nombre_modelo)

    atributos = metadatos["atributos"]
    objetivo = metadatos["objetivo"]
    algoritmo = metadatos["algoritmo"]

    faltantes = [attr for attr in atributos if attr not in valores]
    if faltantes:
        raise ValueError(
            "Faltan valores para: " + ", ".join(faltantes)
        )

    fila = {}
    for atributo in atributos:
        texto = str(valores[atributo]).strip()
        if texto == "":
            raise ValueError(f"Debes ingresar un valor para '{atributo}'.")
        fila[atributo] = _convertir_valor(df, atributo, texto)

    muestra = pd.DataFrame([fila], columns=atributos)
    prediccion = modelo.predict(muestra)[0]

    probabilidades = []
    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(muestra)[0]
        for clase, probabilidad in zip(modelo.classes_, probs):
            probabilidades.append({
                "clase": str(clase),
                "probabilidad": float(probabilidad),
            })
        probabilidades.sort(key=lambda item: item["probabilidad"], reverse=True)

    explicacion = {
        "prediccion": str(prediccion),
        "algoritmo": algoritmo,
        "valores": {clave: fila[clave] for clave in atributos},
        "descripcion_planta": DESCRIPCIONES_PLANTAS.get(str(prediccion).lower()),
        "resumen": _resumen_prediccion(str(prediccion), algoritmo),
        "razones": _explicar_por_estadisticas(
            df, atributos, objetivo, prediccion, fila
        ),
        "probabilidades": probabilidades,
        "pasos_modelo": [],
        "vecinos": [],
    }

    if algoritmo == "id3":
        explicacion["pasos_modelo"] = _explicar_id3(modelo, muestra, atributos)
    elif algoritmo == "knn":
        explicacion["vecinos"] = _explicar_knn(
            modelo, muestra, atributos, metadatos.get("parametros", {})
        )

    try:
        explicacion["explicacion_gemini"] = explicar_como_experto(explicacion)
        explicacion["error_gemini"] = None
    except Exception as error:
        explicacion["explicacion_gemini"] = None
        explicacion["error_gemini"] = str(error)

    return explicacion


def _ruta_metadatos(nombre_modelo):
    if not nombre_modelo.endswith(".joblib"):
        nombre_modelo += ".joblib"
    nombre_json = nombre_modelo.replace(".joblib", ".json")
    return os.path.join(MODEL_FOLDER, nombre_json)


def _convertir_valor(df, atributo, texto):
    if atributo in df.select_dtypes(include="number").columns:
        try:
            return float(texto.replace(",", "."))
        except ValueError:
            raise ValueError(
                f"El valor de '{atributo}' debe ser numérico."
            )
    return texto


def _resumen_prediccion(clase, algoritmo):
    nombre_algoritmo = "ID3" if algoritmo == "id3" else "K-NN"
    return (
        f"Según el modelo {nombre_algoritmo}, la planta pertenece a la clase "
        f"«{clase}» porque las medidas ingresadas coinciden con el patrón "
        f"aprendido para esa especie."
    )


def _explicar_por_estadisticas(df, atributos, objetivo, clase, valores):
    razones = []
    if objetivo not in df.columns:
        return razones

    grupo = df[df[objetivo].astype(str) == str(clase)]
    if grupo.empty:
        return razones

    for atributo in atributos:
        if atributo not in grupo.columns:
            continue
        if not pd.api.types.is_numeric_dtype(grupo[atributo]):
            continue

        valor = float(valores[atributo])
        media = float(grupo[atributo].mean())
        minimo = float(grupo[atributo].min())
        maximo = float(grupo[atributo].max())
        diferencia = abs(valor - media)

        if minimo <= valor <= maximo:
            cercania = "muy cercano" if diferencia <= (maximo - minimo) * 0.25 else "dentro"
            razones.append(
                f"El valor de {atributo} ({_fmt(valor)}) está {cercania} al rango "
                f"típico de {clase} ({_fmt(minimo)} a {_fmt(maximo)}; "
                f"promedio {_fmt(media)})."
            )
        else:
            razones.append(
                f"El valor de {atributo} ({_fmt(valor)}) queda fuera del rango "
                f"usual de {clase} ({_fmt(minimo)} a {_fmt(maximo)}), "
                f"pero otras características empujan la clasificación hacia esa planta."
            )

    return razones


def _explicar_id3(modelo, muestra, atributos):
    arbol = modelo.tree_
    nodo = 0
    pasos = []
    fila = muestra.iloc[0]

    while arbol.feature[nodo] != _tree.TREE_UNDEFINED:
        indice = arbol.feature[nodo]
        nombre = atributos[indice]
        umbral = float(arbol.threshold[nodo])
        valor = float(fila[nombre])

        if valor <= umbral:
            pasos.append(
                f"{nombre} = {_fmt(valor)} ≤ {_fmt(umbral)}, "
                "se toma la rama izquierda."
            )
            nodo = arbol.children_left[nodo]
        else:
            pasos.append(
                f"{nombre} = {_fmt(valor)} > {_fmt(umbral)}, "
                "se toma la rama derecha."
            )
            nodo = arbol.children_right[nodo]

    clase_hoja = modelo.classes_[int(np.argmax(arbol.value[nodo]))]
    pasos.append(
        f"Se llega a una hoja del árbol cuya clase mayoritaria es {clase_hoja}."
    )
    return pasos


def _explicar_knn(modelo, muestra, atributos, parametros):
    k = int(parametros.get("n_neighbors", getattr(modelo, "n_neighbors", 5)))
    distancias, indices = modelo.kneighbors(muestra, n_neighbors=k)
    vecinos = []
    X_train = modelo._fit_X
    y_train = modelo._y
    clases = modelo.classes_

    for distancia, indice in zip(distancias[0], indices[0]):
        clase = clases[int(y_train[indice])]
        valores = {}
        for i, nombre in enumerate(atributos):
            valores[nombre] = _fmt(float(X_train[indice, i]))
        vecinos.append({
            "clase": str(clase),
            "distancia": float(distancia),
            "valores": valores,
        })
    return vecinos


def _fmt(numero):
    return f"{numero:.2f}".rstrip("0").rstrip(".")
