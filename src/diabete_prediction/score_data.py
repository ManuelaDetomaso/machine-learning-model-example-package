import logging
import mlflow.pyfunc
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Dict, Optional


from diabete_prediction.config_loader import load_config

logger = logging.getLogger(__name__)


class ModelScorer:

    def __init__(self):
        self.config = load_config()
        self.feature_schema = self.config["InputData"]["feature_schema"]
        self.feature_names = [feat["name"] for feat in self.feature_schema]
        self.predictions_table_name = self.config["OutputData"][
            "predictions_table_name"
        ]

    def _get_predict_udf(self, model_uri: str) -> pd.Series:
        """
        Create a Pandas UDF for scoring using an MLflow model.

        Args:
            model_uri (str): Model URI (e.g., models:/<model-name>/<version>)

        Returns:
            A PySpark Pandas UDF that applies the model's prediction.
        """
        logger.info("🔍 Loading MLflow model from URI: %s", model_uri)
        model = mlflow.pyfunc.load_model(model_uri)

        @pandas_udf(FloatType())
        def predict_udf(*cols):
            X = pd.concat(cols, axis=1)
            X.columns = self.feature_names
            logger.debug("🧪 Predicting using model on batch of shape: %s", X.shape)
            preds = model.predict(X)
            return pd.Series(preds)

        logger.info("✅ Model UDF created successfully.")
        return predict_udf

    def get_model_versions_by_experiment_prefix(
        self,
        prefix: str,
        return_latest_only: bool = False,
        max_runs_per_experiment: int = 100,
    ) -> Dict[str, List[Dict]]:
        """
        Returns model versions associated with experiments starting with a given prefix.

        Args:
            prefix (str): Prefix to match experiment names.
            return_latest_only (bool): If True, returns only the latest model version per experiment.
            max_runs_per_experiment (int): Max number of recent runs to check per experiment.

        Returns:
            Dict[str, List[Dict]]: Dict mapping experiment name to list of model versions found.
        """
        client = MlflowClient()
        model_versions = {}

        # Find matching experiments
        experiments = mlflow.search_experiments()
        matching_experiments = [exp for exp in experiments if exp.name.startswith(prefix)]

        for exp in matching_experiments:
            exp_versions = []
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=max_runs_per_experiment,
            )

            for _, run in runs.iterrows():
                run_id = run["run_id"]
                versions = client.search_model_versions(f"run_id = '{run_id}'")
                # return metadata for each experiment
                for v in versions:
                    exp_versions.append({
                        "model_name": v.name,
                        "version": int(v.version),
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                    })

            # Sort versions (optional)
            exp_versions = sorted(exp_versions, key=lambda x: x["version"], reverse=True)

            if return_latest_only and exp_versions:
                model_versions[exp.name] = [exp_versions[0]]
            else:
                model_versions[exp.name] = exp_versions

        return model_versions

    def generate_predictions_dataframe(
        self,
        df_inference: DataFrame,
        experiment_name,
        model_type="regression",
        model_version=None,
        save=True,
    ) -> DataFrame:
        """
        Generate predictions using the trained model and optionally save them.

        Args:
            df_inference (DataFrame): Input PySpark DataFrame with features
            experiment_name (str): MLflow experiment name (used to construct model URI), use: 
                config["ModelGeneral"]["experiment"]
            model_type (str): Model type (regression or classification_binary)
            model_version (None): Version of the registered model, if None, the last version is retrieved
            save (bool): Whether to save the resulting DataFrame to a Delta table

        Returns:
            DataFrame: DataFrame with predictions column added
        """
        # if model version is None, use the last available
        if model_version is None:
            #get models with searche experiment name and their last versions
            latest_versions = self.get_model_versions_by_experiment_prefix(
                experiment_name, return_latest_only=True
            )
            model_to_use = f"{experiment_name}-{model_type}"
            model_version = latest_versions[model_to_use][0]["version"]

        model_uri = f"models:/{experiment_name}-{model_type}-model/{model_version}"
        logger.info("🚀 Generating predictions using model: %s", model_uri)

        _predict_udf = self._get_predict_udf(model_uri)

        df_preds = df_inference.withColumn(
            "predictions",
            _predict_udf(*[df_inference[col] for col in self.feature_names]),
        )

        logger.info("✅ Predictions column added to DataFrame.")

        if save:
            logger.info(
                "💾 Saving predictions to Delta table: %s", self.predictions_table_name
            )
            df_preds.write.format("delta").mode("overwrite").option(
                "mergeSchema", "true"
            ).saveAsTable(self.predictions_table_name)
            logger.info("✅ Predictions saved successfully.")

        return df_preds
