import logging
import mlflow.pyfunc
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

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

    def generate_predictions_dataframe(
        self,
        df_inference: DataFrame,
        experiment_name,
        model_type="regression",
        model_version=1,
        save=True,
    ) -> DataFrame:
        """
        Generate predictions using the trained model and optionally save them.

        Args:
            df_inference (DataFrame): Input PySpark DataFrame with features
            experiment_name (str): MLflow experiment name (used to construct model URI)
            model_type (str): Model type (regression or classification_binary)
            model_version (int): Version of the registered model
            save (bool): Whether to save the resulting DataFrame to a Delta table

        Returns:
            DataFrame: DataFrame with predictions column added
        """
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
