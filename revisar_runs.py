import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

experiment = mlflow.get_experiment_by_name(
    "Iris_Classification"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"]
)

best_run = runs.iloc[0]

print("Mejor modelo:")
print("Run ID:", best_run["run_id"])
print("Accuracy:", best_run["metrics.accuracy"])

model_uri = f"runs:/{best_run['run_id']}/model"

print("Model URI:", model_uri)