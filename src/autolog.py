import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sys.stdout.reconfigure(encoding='utf-8')

import dagshub
dagshub.init(repo_owner='manoharnimiditalli001', repo_name='mlflow', mlflow=True)
# Fix Unicode issue
os.environ["PYTHONIOENCODING"] = "utf-8"

# Patch MLflow to remove emoji logs
import mlflow.tracking._tracking_service.client as mlflow_client
def patched_log_url(self, run_id):
    run = self.get_run(run_id)
    run_name = run.data.tags.get("mlflow.runName", run_id)
    run_url = f"{mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run_id}"
    sys.stdout.write(f"View run {run_name} at: {run_url}\n")

mlflow_client.TrackingServiceClient._log_url = patched_log_url

# Set MLflow tracking URI
mlflow.autolog()
mlflow.set_tracking_uri("https://dagshub.com/manoharnimiditalli001/mlflow.mlflow")

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define parameters
max_depth = 5
n_estimators = 5

mlflow.set_experiment("mlflow1")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion-matrix.png")

    # Log artifacts
    
    # Log script file
    if hasattr(sys, "argv"):
        mlflow.log_artifact(sys.argv[0])

    # Set tags
    mlflow.set_tags({"Author": "manohar", "Project": "Wine Classification"})

    # Log model
    

    print(f"Model Accuracy: {accuracy:.4f}")
