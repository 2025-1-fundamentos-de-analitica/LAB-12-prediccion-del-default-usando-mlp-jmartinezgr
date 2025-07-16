# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
"""
Modelo de predicción de default para clientes de tarjetas de crédito.

Este módulo implementa un pipeline completo de machine learning para predecir
el default de pago del próximo mes basado en 23 variables explicativas.
"""

import os
import json
import gzip
import time
import pickle
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# Configuración
warnings.filterwarnings("ignore", category=UserWarning)

# Constantes
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
TARGET_COLUMN = "default"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"
TRAIN_DATA_PATH = "files/input/train_data.csv.zip"
TEST_DATA_PATH = "files/input/test_data.csv.zip"


class DataProcessor:
    """Clase para el procesamiento y limpieza de datos."""

    @staticmethod
    def clean_data(data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos según las especificaciones del proyecto.

        Args:
            data_df: DataFrame con los datos originales

        Returns:
            DataFrame limpio
        """
        df = data_df.copy()

        # Renombrar columna objetivo
        df = df.rename(columns={"default payment next month": TARGET_COLUMN})

        # Remover columna ID
        df = df.drop(columns="ID")

        # Recodificar variables categóricas (0 -> NaN)
        df["EDUCATION"] = df["EDUCATION"].replace(0, np.nan)
        df["MARRIAGE"] = df["MARRIAGE"].replace(0, np.nan)

        # Eliminar registros con información faltante
        df = df.dropna()

        # Agrupar valores de EDUCATION > 4 en categoría "others"
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

        return df

    @staticmethod
    def split_features_target(
        data: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa características y variable objetivo.

        Args:
            data: DataFrame con todos los datos
            target_column: Nombre de la columna objetivo

        Returns:
            Tupla con (features, target)
        """
        X = data.drop(columns=target_column)
        y = data[target_column]
        return X, y

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga y limpia los datos de entrenamiento y prueba.

        Returns:
            Tupla con (train_data, test_data) limpios
        """
        # Cargar datos
        train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=False, compression="zip")
        test_data = pd.read_csv(TEST_DATA_PATH, index_col=False, compression="zip")

        # Limpiar datos
        train_data = DataProcessor.clean_data(train_data)
        test_data = DataProcessor.clean_data(test_data)

        return train_data, test_data


class ModelPipeline:
    """Clase para crear y configurar el pipeline de machine learning."""

    @staticmethod
    def create_pipeline(df: pd.DataFrame) -> Pipeline:
        """
        Crea el pipeline de procesamiento y modelado.

        Args:
            df: DataFrame para determinar características numéricas

        Returns:
            Pipeline configurado
        """
        # Identificar características numéricas
        numerical_features = [
            col for col in df.columns if col not in CATEGORICAL_FEATURES
        ]

        # Configurar preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), CATEGORICAL_FEATURES),
            ],
            remainder=StandardScaler(),
        )

        # Crear pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("select_k_best", SelectKBest(f_classif)),
                ("pca", PCA()),
                ("model", MLPClassifier(max_iter=15000)),
            ]
        )

        return pipeline

    @staticmethod
    def get_hyperparameter_grid() -> Dict[str, Any]:
        """
        Define la grilla de hiperparámetros para optimización.

        Returns:
            Diccionario con los hiperparámetros a optimizar
        """
        return {
            "pca__n_components": [20],
            "select_k_best__k": [20],
            "model__hidden_layer_sizes": [(35, 35, 30, 30, 30, 30, 30, 30)],
            "model__activation": ["relu"],
            "model__solver": ["adam"],
            "model__alpha": [0.353],
            "model__learning_rate_init": [0.0005],
        }


class ModelTrainer:
    """Clase para entrenar y optimizar el modelo."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.best_model = None

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> GridSearchCV:
        """
        Optimiza los hiperparámetros usando validación cruzada.

        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            GridSearchCV con el mejor modelo
        """
        param_grid = ModelPipeline.get_hyperparameter_grid()

        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=10,
            scoring="balanced_accuracy",
            verbose=1,
            n_jobs=-1,
        )

        with mlflow.start_run():
            grid_search.fit(X_train, y_train)

            # Logging de MLflow
            self._log_mlflow_metrics(grid_search, X_train, y_train)

        self.best_model = grid_search
        return grid_search

    def _log_mlflow_metrics(
        self, grid_search: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series
    ):
        """
        Registra métricas y parámetros en MLflow.

        Args:
            grid_search: Objeto GridSearchCV entrenado
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
        """
        # Log parámetros
        best_params = grid_search.best_params_
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Log métricas
        y_pred_train = grid_search.predict(X_train)
        metrics = {
            "precision": precision_score(y_train, y_pred_train),
            "balanced_accuracy": balanced_accuracy_score(y_train, y_pred_train),
            "recall": recall_score(y_train, y_pred_train),
            "f1_score": f1_score(y_train, y_pred_train),
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log modelo
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")


class ModelEvaluator:
    """Clase para evaluar el modelo y calcular métricas."""

    @staticmethod
    def calculate_metrics(
        model: GridSearchCV,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Dict, Dict]:
        """
        Calcula métricas para conjuntos de entrenamiento y prueba.

        Args:
            model: Modelo entrenado
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba

        Returns:
            Tupla con (métricas_train, métricas_test)
        """
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics_train = {
            "type": "metrics",
            "dataset": "train",
            "precision": float(precision_score(y_train, y_train_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
            "recall": float(recall_score(y_train, y_train_pred)),
            "f1_score": float(f1_score(y_train, y_train_pred)),
        }

        metrics_test = {
            "type": "metrics",
            "dataset": "test",
            "precision": float(precision_score(y_test, y_test_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
            "recall": float(recall_score(y_test, y_test_pred)),
            "f1_score": float(f1_score(y_test, y_test_pred)),
        }

        return metrics_train, metrics_test

    @staticmethod
    def calculate_confusion_matrices(
        model: GridSearchCV,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Dict, Dict]:
        """
        Calcula matrices de confusión para conjuntos de entrenamiento y prueba.

        Args:
            model: Modelo entrenado
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba

        Returns:
            Tupla con (cm_train, cm_test)
        """
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        cm_matrix_train = {
            "type": "cm_matrix",
            "dataset": "train",
            "true_0": {
                "predicted_0": int(cm_train[0, 0]),
                "predicted_1": int(cm_train[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm_train[1, 0]),
                "predicted_1": int(cm_train[1, 1]),
            },
        }

        cm_matrix_test = {
            "type": "cm_matrix",
            "dataset": "test",
            "true_0": {
                "predicted_0": int(cm_test[0, 0]),
                "predicted_1": int(cm_test[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm_test[1, 0]),
                "predicted_1": int(cm_test[1, 1]),
            },
        }

        return cm_matrix_train, cm_matrix_test


class ModelPersistence:
    """Clase para guardar y cargar modelos."""

    @staticmethod
    def save_model(model: GridSearchCV, filepath: str = MODEL_PATH):
        """
        Guarda el modelo comprimido.

        Args:
            model: Modelo a guardar
            filepath: Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Guardar modelo comprimido
        with gzip.open(filepath, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(filepath: str = MODEL_PATH) -> GridSearchCV:
        """
        Carga el modelo desde archivo.

        Args:
            filepath: Ruta del modelo

        Returns:
            Modelo cargado
        """
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_metrics(metrics_list: List[Dict], filepath: str = METRICS_PATH):
        """
        Guarda las métricas en formato JSON.

        Args:
            metrics_list: Lista de diccionarios con métricas
            filepath: Ruta donde guardar las métricas
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Guardar métricas
        pd.DataFrame(metrics_list).to_json(filepath, orient="records", lines=True)


class DefaultPredictionPipeline:
    """Pipeline principal para la predicción de default."""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = None
        self.model_evaluator = ModelEvaluator()
        self.model_persistence = ModelPersistence()

    def run_complete_pipeline(self):
        """
        Ejecuta el pipeline completo de entrenamiento y evaluación.
        """
        print("🔄 Iniciando pipeline de predicción de default...")

        # 1. Cargar y procesar datos
        print("📊 Cargando y procesando datos...")
        train_data, test_data = self.data_processor.load_data()

        # 2. Separar características y objetivo
        X_train, y_train = self.data_processor.split_features_target(
            train_data, TARGET_COLUMN
        )
        X_test, y_test = self.data_processor.split_features_target(
            test_data, TARGET_COLUMN
        )

        # 3. Crear pipeline
        print("🏗️ Creando pipeline de ML...")
        pipeline = ModelPipeline.create_pipeline(X_train)

        # 4. Entrenar modelo
        print("🚀 Entrenando modelo...")
        self.model_trainer = ModelTrainer(pipeline)
        start_time = time.time()

        model = self.model_trainer.optimize_hyperparameters(X_train, y_train)

        end_time = time.time()
        print(f"⏱️ Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")
        print(f"🎯 Mejores parámetros: {model.best_params_}")

        # 5. Guardar modelo
        print("💾 Guardando modelo...")
        self.model_persistence.save_model(model)

        # 6. Evaluar modelo
        print("📈 Evaluando modelo...")
        metrics_train, metrics_test = self.model_evaluator.calculate_metrics(
            model, X_train, y_train, X_test, y_test
        )

        cm_train, cm_test = self.model_evaluator.calculate_confusion_matrices(
            model, X_train, y_train, X_test, y_test
        )

        # 7. Guardar métricas
        print("📋 Guardando métricas...")
        all_metrics = [metrics_train, metrics_test, cm_train, cm_test]
        self.model_persistence.save_metrics(all_metrics)

        # 8. Mostrar resultados
        print("\n📊 Resultados:")
        print("Entrenamiento:", metrics_train)
        print("Prueba:", metrics_test)
        print("Matriz de confusión - Entrenamiento:", cm_train)
        print("Matriz de confusión - Prueba:", cm_test)

        print("✅ Pipeline completado exitosamente!")

        return model


def main():
    """Función principal para ejecutar el pipeline."""
    pipeline = DefaultPredictionPipeline()
    model = pipeline.run_complete_pipeline()
    return model


if __name__ == "__main__":
    main()
