import mlflow.pyfunc
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

from diabete_prediction.config_loader import load_config


class ModelScorer:

    def __init__(self):
        self.config = load_config()
        self.feature_schema = self.config["InputData"]["feature_schema"]
        self.feature_names = [feat["name"] for feat in self.feature_schema]
        self.predictions_table_name = self.config["OutputData"][
            "predictions_table_name"
        ]

    def _get_predict_udf(self, model_uri: str) -> pd.Series:
        """_summary_

        Args:
            model_uri (str): model uri (relative path to mlflow model based on its registered name)

        Returns:
            pd.Series: pandas Series, i.e. the pandas dataframe column with predictions
        """
        model = mlflow.pyfunc.load_model(model_uri)

        @pandas_udf(FloatType())
        def predict_udf(*cols):
            X = pd.concat(cols, axis=1)
            X.columns = self.feature_names
            preds = model.predict(X)
            return pd.Series(preds)

        return predict_udf

    def generate_predictions_dataframe(
        self,
        df_inference: DataFrame,
        experiment_name,
        model_type="regression",
        model_version=1,
        save=True,
    ) -> DataFrame:
        """Return prepared inference data with new columns containing predictions

        Args:
            df_inference (DataFrame): prepared inference data
            experiment_name (str, optional): name of mlflow experiment.
            model_type (str, optional): model type: either regression or classification_binary. Defaults to "regression".
            model_version (int, optional): mlflow model registered version. Defaults to 1.
            save (bool, optional): whether to save the dataset with predicitons or not. Defaults to True.

        Returns:
            DataFrame: prepared inference data with new columns containing predictions.
        """
        # Use the model to generate diabetes predictions for each row
        model_uri = f"models:/{experiment_name}-{model_type}-model/{model_version}"
        _predict_udf = self._get_predict_udf(model_uri)

        print("Uploading {}".format(model_uri))
        df_preds = df_inference.withColumn(
            "predictions",
            _predict_udf(*[df_inference[col] for col in self.feature_names]),
        )

        print("Prepared inference data with new columns containing predictions.")
        if save:
            # Save the results (the original features PLUS the prediction)
            df_preds.write.format("delta").mode("overwrite").option(
                "mergeSchema", "true"
            ).saveAsTable(self.predictions_table_name)
            print("Saved prepared inference data with predictions.")
        return df_preds
