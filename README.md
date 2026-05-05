# ml-pipeline-final-Nicolas-Guerrero

# Automatización de un Pipeline de Machine Learning con GitHub Actions y MLflow

## Objetivo

Este proyecto implementa un pipeline reproducible de Machine Learning que permite cargar datos, preprocesarlos, entrenar un modelo, evaluarlo, registrar métricas y guardar el modelo usando MLflow. El pipeline se automatiza con GitHub Actions.

## Dataset

Se utiliza el dataset externo Pima Indians Diabetes, disponible públicamente en:

https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

Este dataset se usa para predecir si una persona presenta diabetes según variables clínicas. No se utilizan datasets de `sklearn.datasets`.

## Estructura del proyecto

```text
.
├── src/
│   ├── train.py
│   └── test_pipeline.py
├── config.yaml
├── requirements.txt
├── Makefile
├── mlruns/
├── models/
├── reports/
└── .github/
    └── workflows/
        └── ml.yml
