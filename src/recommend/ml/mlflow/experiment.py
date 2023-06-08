import sqlite3
import os
import mlflow

from src.recommend.ml.imf.trainer import Trainer
from src.recommend.ml.imf.fetcher import fetch_data_for_train
from src.recommend.ml.mlflow.config import EXPERIMENT_NAME, parameter

DB_PATH = "./src/recommend/ml/mlflow/sqlite/mlruns.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
# バックエンド用DBの作成
conn = sqlite3.connect(DB_PATH)

tracking_uri = f"sqlite:///{DB_PATH}"
mlflow.set_tracking_uri(tracking_uri)

ARTIFACT_LOCATION = "./src/recommend/ml/mlflow/artifact"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION
    )
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_params(parameter)
    trainer = Trainer(**parameter)
    raw_df = fetch_data_for_train()
    metrics = trainer.execute_experiment(raw_df=raw_df)
    mlflow.log_metric("precision_at_k", metrics.precision_at_k)
    mlflow.log_metric("recall_at_k", metrics.recall_at_k)
