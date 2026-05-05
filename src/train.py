import json
import os
import sys
import traceback

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(config):
    dataset_config = config["dataset"]

    df = pd.read_csv(
        dataset_config["url"],
        names=dataset_config["columns"],
    )

    return df


def preprocess_data(df, config):
    target_column = config["dataset"]["target_column"]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # En el dataset Pima, algunos ceros representan datos faltantes.
    columns_with_invalid_zero = [
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
    ]

    X[columns_with_invalid_zero] = X[columns_with_invalid_zero].replace(0, np.nan)

    return X, y


def build_model(config):
    model_config = config["model"]

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=model_config["max_iter"],
                    random_state=model_config["random_state"],
                ),
            ),
        ]
    )

    return pipeline


def main():
    try:
        print("Iniciando pipeline de Machine Learning...")

        config = load_config()

        os.makedirs("mlruns", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        tracking_uri = "file://" + os.path.abspath(config["mlflow"]["tracking_dir"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        print(f"MLflow Tracking URI: {tracking_uri}")

        df = load_dataset(config)
        X, y = preprocess_data(df, config)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["model"]["test_size"],
            random_state=config["model"]["random_state"],
            stratify=y,
        )

        model = build_model(config)

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_param("dataset_url", config["dataset"]["url"])
            mlflow.log_param("dataset_name", "Pima Indians Diabetes")
            mlflow.log_param("model_type", config["model"]["type"])
            mlflow.log_param("test_size", config["model"]["test_size"])
            mlflow.log_param("random_state", config["model"]["random_state"])
            mlflow.log_param("max_iter", config["model"]["max_iter"])

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "run_id": run_id,
            }

            with open(config["paths"]["metrics_output"], "w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=4)

            report = classification_report(y_test, y_pred)

            with open(config["paths"]["report_output"], "w", encoding="utf-8") as file:
                file.write(report)

            joblib.dump(model, config["paths"]["model_output"])

            input_example = X_test.head(5)
            signature = infer_signature(input_example, model.predict(input_example))

            mlflow.log_artifact(config["paths"]["metrics_output"])
            mlflow.log_artifact(config["paths"]["report_output"])
            mlflow.log_artifact(
                config["paths"]["model_output"], artifact_path="model_file"
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            print("Pipeline ejecutado correctamente.")
            print(f"Run ID: {run_id}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Modelo guardado en: {config['paths']['model_output']}")
            print("Modelo registrado en MLflow correctamente.")

    except Exception:
        print("Error durante la ejecución del pipeline.")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
