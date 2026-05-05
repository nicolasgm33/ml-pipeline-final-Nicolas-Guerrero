import os
import sys

import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    dataset_url = config["dataset"]["url"]
    columns = config["dataset"]["columns"]
    target_column = config["dataset"]["target_column"]

    df = pd.read_csv(dataset_url, names=columns)

    assert not df.empty, "El dataset no puede estar vacío."
    assert target_column in df.columns, "La columna objetivo no existe."
    assert len(df.columns) == len(columns), "Las columnas no coinciden."
    assert (
        "sklearn.datasets" not in dataset_url
    ), "No se permite usar datasets de sklearn.datasets."

    os.makedirs("reports", exist_ok=True)

    with open("reports/test_results.txt", "w", encoding="utf-8") as file:
        file.write("Pruebas básicas ejecutadas correctamente.\n")
        file.write(f"Filas del dataset: {len(df)}\n")
        file.write(f"Columnas del dataset: {len(df.columns)}\n")

    print("Pruebas básicas ejecutadas correctamente.")
    print(f"Dataset externo validado: {dataset_url}")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"Error de validación: {error}")
        sys.exit(1)
