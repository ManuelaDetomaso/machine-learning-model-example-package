import logging

import mlflow
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from pyspark.sql import DataFrame

from diabete_prediction.config_loader import load_config
from diabete_prediction.utils import create_mlflow_schema_from_typed_list

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.config = load_config()
        self.risk_colname = self.config["InputData"]["risk_colname"]
        self.target_colname = self.config["InputData"]["target_colname"]
        self.feature_schema = self.config["InputData"]["feature_schema"]
        self.feature_names = [feat["name"] for feat in self.feature_schema]
        self.test_size = self.config["ModelTraining"]["test_size"]
        self.experiment_name = self.config["ModelGeneral"]["experiment_name"]

    def split_train_test_data(
        self, df_sp_prepared: DataFrame, target_type="numerical"
    ) -> tuple:
        """Split prepared training data into the feature set and the target variable"""
        from sklearn.model_selection import train_test_split

        df_prepared = df_sp_prepared.coalesce(1).toPandas()

        if target_type == "numerical":
            X, y = (
                df_prepared[self.feature_names].values,
                df_prepared[self.target_colname].values,
            )
        elif target_type == "categorical":
            X, y = (
                df_prepared[self.feature_names].values,
                df_prepared[self.risk_colname].values,
            )
        else:
            logger.error(
                "Invalid target_type '%s'. Must be 'numerical' or 'categorical'.",
                target_type,
            )
            raise ValueError("Indicate a target type, either numerical or categorical")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=0
        )

        logger.info("🔀 Data split complete.")
        logger.debug(
            "X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape
        )
        logger.debug(
            "y_train shape: %s | y_test shape: %s", y_train.shape, y_test.shape
        )

        return X_train, X_test, y_train, y_test

    def mlflow_training(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "regression",
        with_signature: bool = True,
        logs: bool = False,
        save_model=True,
    ):
        """Train an ML model and optionally log to MLflow"""
        from sklearn.linear_model import LinearRegression, LogisticRegression

        if model_type == "regression":
            output_schema = Schema([ColSpec("integer")])
            full_experiment_name = self.experiment_name + "-regression"
            model = LinearRegression()
            logger.info("📈 Initialized LinearRegression model.")
        elif model_type == "categorical_binary":
            output_schema = Schema([ColSpec("binary")])
            full_experiment_name = self.experiment_name + "-classification"
            model = LogisticRegression(C=1 / 0.1, solver="liblinear")
            logger.info("📊 Initialized LogisticRegression model.")
        else:
            logger.error("Unsupported model type: %s", model_type)
            raise ValueError(
                "A model type can be either regression or classification_binary"
            )

        mlflow.set_experiment(full_experiment_name)

        with mlflow.start_run():
            mlflow.autolog(log_models=logs)
            model.fit(X_train, y_train)
            logger.info("✅ Model training completed.")

            if with_signature:
                logger.info("🧾 Applying MLflow model signature.")
                input_schema = create_mlflow_schema_from_typed_list(self.feature_schema)
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                mlflow.sklearn.log_model(model, "model", signature=signature)
                logger.info("📁 Model logged to MLflow with signature.")

            if save_model:
                self.save_mlflow_model(full_experiment_name)

    def save_mlflow_model(self, experiment_name: str):
        """Save and register an MLflow model from the latest run of an experiment"""
        model_name = experiment_name + "-model"
        exp = mlflow.get_experiment_by_name(experiment_name)
        last_run = mlflow.search_runs(
            exp.experiment_id, order_by=["start_time DESC"], max_results=1
        )
        
        last_run_id = last_run.iloc[0]["run_id"]
        logger.info("💾 Registering model from run ID: %s", last_run_id)

        model_uri = f"runs:/{last_run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)

        logger.info("✅ Model registered: Name=%s | Version=%s", mv.name, mv.version)
