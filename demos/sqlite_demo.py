import os
import mlflow

root_dir = os.getcwd()
database_dir = "mlruns.db"
tracking_uri = "sqlite:///" + root_dir + database_dir
mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run():
    mlflow.log_metric("aaa", 1)
    mlflow.log_param("bbb", 2)