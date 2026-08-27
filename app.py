from dotenv import load_dotenv
from flask import Flask, render_template, request

load_dotenv()

from services.dataset_service import (
    obtener_datasets,
    cargar_dataset,
    obtener_informacion,
    obtener_preview
)

from services.training_service import (
    entrenar_modelo,
    guardar_modelo
)

from services.prediction_service import (
    guardar_metadatos_modelo,
    predecir_muestra
)

from services.visualization_service import (
    generar_matriz_confusion,
    generar_arbol_id3,
    generar_grafica_knn
)

app = Flask(__name__)

# ============================================================
# RUTA PRINCIPAL
# ============================================================

@app.route("/", methods=["GET", "POST"])
def inicio():

    # ========================================================
    # VARIABLES INICIALES
    # ========================================================
    datasets = obtener_datasets()
    informacion = None
    preview = None
    dataset_seleccionado = None
    columnas = []
    columnas_seleccionadas = []
    columna_objetivo = None
    algoritmo = None
    resultados = None
    error = None
    nombre_modelo = None
    prediccion = None
    valores_prueba = {}

    # ========================================================
    # PROCESAR FORMULARIO
    # ========================================================
    if request.method == "POST":

        # ----------------------------------------------------
        # Obtener información enviada por el formulario
        # ----------------------------------------------------
        dataset_seleccionado = request.form.get("dataset")
        columnas_seleccionadas = request.form.getlist("atributos")
        columna_objetivo = request.form.get("objetivo")
        algoritmo = request.form.get("algoritmo")
        
        # NUEVO: Detectar si se presionó el botón de entrenar
        btn_entrenar = request.form.get("btn_entrenar")
        btn_predecir = request.form.get("btn_predecir")
        nombre_modelo = request.form.get("nombre_modelo") 

        try:
            # =================================================
            # 1. CARGAR DATASET Y OBTENER COLUMNAS (Siempre que haya dataset)
            # =================================================
            if dataset_seleccionado:
                df = cargar_dataset(dataset_seleccionado)
                informacion = obtener_informacion(df)
                preview = obtener_preview(df).to_html(classes="tabla-dataset", index=False)
                columnas = df.columns.tolist()

            # =================================================
            # 5. ENTRENAMIENTO (Solo si se presionó el botón)
            # =================================================
            if btn_entrenar:
                
                if not dataset_seleccionado:
                    raise ValueError("Debes seleccionar un dataset.")

                if not algoritmo:
                    raise ValueError("Debes seleccionar un algoritmo.")

                if not columnas_seleccionadas:
                    raise ValueError("Debes seleccionar al menos un atributo.")

                if not columna_objetivo:
                    raise ValueError("Debes seleccionar una variable objetivo.")

                if columna_objetivo in columnas_seleccionadas:
                    raise ValueError("La variable objetivo no puede ser también un atributo.")

                # =================================================
                # 6. CREAR X E Y
                # =================================================
                X = df[columnas_seleccionadas]
                y = df[columna_objetivo]

                # =================================================
                # 7. HOLD-OUT
                # =================================================
                holdout = request.form.get("holdout", "0.30")
                try:
                    test_size = float(holdout)
                except ValueError:
                    raise ValueError("El porcentaje de prueba no es válido.")

                # =================================================
                # 8. PARÁMETROS
                # =================================================
                parametros = {}

                if algoritmo == "id3":
                    parametros["max_depth"] = request.form.get("max_depth", "")
                elif algoritmo == "knn":
                    parametros["n_neighbors"] = request.form.get("n_neighbors", "5")
                    parametros["weights"] = request.form.get("weights", "uniform")
                    parametros["metric"] = request.form.get("metric", "minkowski")

                # =================================================
                # 9. ENTRENAR MODELO
                # =================================================
                modelo, resultados = entrenar_modelo(X, y, algoritmo, test_size, parametros)

                # =================================================
                # 10. GENERAR MATRIZ DE CONFUSIÓN
                # =================================================
                X_test = resultados["X_test"]
                y_test = resultados["y_test"]

                ruta_confusion = generar_matriz_confusion(
                    modelo, X_test, y_test, f"confusion_{algoritmo}.png"
                )
                resultados["confusion_image"] = ruta_confusion.replace("\\", "/").replace("static/", "")

                # =================================================
                # 11. GENERAR ÁRBOL ID3
                # =================================================
                if algoritmo == "id3":
                    ruta_arbol = generar_arbol_id3(modelo, X, "arbol_id3.png")
                    resultados["tree_image"] = ruta_arbol.replace("\\", "/").replace("static/", "")

                # =================================================
                # GRÁFICA K-NN
                # =================================================
                if algoritmo == "knn":
                    k = int(parametros.get("n_neighbors", 5))
                    ruta_knn = generar_grafica_knn(df, columnas_seleccionadas, columna_objetivo, k, "knn_2d.png")
                    resultados["knn_image"] = ruta_knn.replace("\\", "/").replace("static/", "")

                # =================================================
                # 12. GUARDAR MODELO
                # =================================================
                nombre_dataset = dataset_seleccionado.replace(".csv", "")
                nombre_modelo = f"modelo_{algoritmo}_{nombre_dataset}.joblib"
                ruta_modelo = guardar_modelo(modelo, nombre_modelo)
                
                resultados["modelo"] = nombre_modelo
                resultados["ruta_modelo"] = ruta_modelo

                guardar_metadatos_modelo(nombre_modelo, {
                    "modelo": nombre_modelo,
                    "algoritmo": algoritmo,
                    "dataset": dataset_seleccionado,
                    "atributos": columnas_seleccionadas,
                    "objetivo": columna_objetivo,
                    "parametros": parametros,
                })

            # =================================================
            # 13. PROBAR MODELO
            # =================================================
            if btn_predecir:
                if not nombre_modelo:
                    raise ValueError(
                        "Primero debes entrenar un modelo para poder probarlo."
                    )

                valores_prueba = {
                    atributo: request.form.get(f"prueba_{atributo}", "")
                    for atributo in columnas_seleccionadas
                }

                prediccion = predecir_muestra(
                    nombre_modelo,
                    valores_prueba,
                    df
                )

        except Exception as e:
            error = str(e)

    # ========================================================
    # RENDERIZAR HTML
    # ========================================================
    return render_template(
        "dataset.html",
        datasets=datasets,
        informacion=informacion,
        preview=preview,
        dataset_seleccionado=dataset_seleccionado,
        columnas=columnas,
        columnas_seleccionadas=columnas_seleccionadas,
        columna_objetivo=columna_objetivo,
        algoritmo=algoritmo,
        resultados=resultados,
        error=error,
        nombre_modelo=nombre_modelo,
        prediccion=prediccion,
        valores_prueba=valores_prueba
    )

# ============================================================
# INICIAR APLICACIÓN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)