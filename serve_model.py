'''

    1.- Ejecutar:
        python3 serve_model.py 
    
    2.- Las puse en un un archivo llamado source "var_env":
            export MLFLOW_TRACKING_URI=http://localhost:5000
            echo $MLFLOW_TRACKING_URI 

            Comandos linux:
                nano var_env 
                cat var_env 
                source var_env 
    
    3.- Registrar las variables de entorno:
            -- En linux:
                source var_env
    4.-
        mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        --host 0.0.0.0 \
        --port 5000


    5.- Probar:
    
        curl -X POST http://localhost:5001/invocations \
          -H "Content-Type: application/json" \
          -d '{
            "dataframe_records": [
              {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2
              }
            ]
          }'

'''

import mlflow
import subprocess
import sys

# --------------------------------------------------
# Configuración
# --------------------------------------------------

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Iris_Classification"
PORT = 5001

# --------------------------------------------------
# Conectar con MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

print(f"MLflow Tracking Server: {TRACKING_URI}")
print(f"Experimento: {EXPERIMENT_NAME}")

# --------------------------------------------------
# Obtener experimento
# --------------------------------------------------

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print(f"ERROR: No existe el experimento '{EXPERIMENT_NAME}'")
    sys.exit(1)

# --------------------------------------------------
# Obtener los runs terminados
# --------------------------------------------------

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"]
)

if runs.empty:
    print("ERROR: No existen runs terminados.")
    sys.exit(1)

# --------------------------------------------------
# Mostrar modelos encontrados
# --------------------------------------------------

print("\nRuns encontrados:")
print("-" * 70)

for _, run in runs.iterrows():

    accuracy = run.get("metrics.accuracy")

    print(
        f"Run: {run['run_id']} | "
        f"Accuracy: {accuracy}"
    )

# --------------------------------------------------
# Seleccionar el mejor modelo
# --------------------------------------------------

best_run = runs.iloc[0]

run_id = best_run["run_id"]
accuracy = best_run["metrics.accuracy"]

model_uri = f"runs:/{run_id}/model"

print("\n" + "=" * 70)
print("MEJOR MODELO")
print("=" * 70)

print(f"Run ID:  {run_id}")
print(f"Accuracy: {accuracy}")
print(f"Model URI: {model_uri}")

# --------------------------------------------------
# Levantar MLflow Model Serving
# --------------------------------------------------

print("\nIniciando MLflow Model Serving...")
print(f"API REST: http://localhost:{PORT}/invocations")
print("Presiona CTRL+C para detener el servicio.\n")

command = [
    "mlflow",
    "models",
    "serve",
    "--model-uri",
    model_uri,
    "--host",
    "0.0.0.0",
    "--port",
    str(PORT),
    "--env-manager",
    "local"
]

subprocess.run(command)