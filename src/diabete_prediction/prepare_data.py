import warnings
import logging

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, FloatType, IntegerType, StringType

from diabete_prediction.config_loader import load_config

logger = logging.getLogger(__name__)


class DataPreparator:
    def __init__(self):
        self.config = load_config()
        self.risk_colname = self.config["InputData"]["risk_colname"]
        self.target_colname = self.config["InputData"]["target_colname"]
        self.target_threshold = self.config["InputData"]["target_threshold"]
        self.expected_target_proportions = self.config["InputData"]["expected_target_proportions"]
        self.feature_schema = self.config["InputData"]["feature_schema"]
        self.prepared_inference_data = self.config["OutputData"]["prepared_inference_data_table_name"]

    def _check_target_labels_proportions(self, df: pd.DataFrame):
        """Checks whether the categorised target values assume expected proportions

        Args:
            df (pd.DataFrame): prepared data
        """
        proportion_dict = dict(
            round(df[self.risk_colname].value_counts() / len(df[self.risk_colname]), 2)
        )
        if proportion_dict != self.expected_target_proportions:
            warnings.warn("Target labels proportions deviate from expectations.")
        else:
            logger.info("✅ Target values proportions are as expected.")

    def prepare_training_data(self, spark_df: DataFrame) -> pd.DataFrame:
        """Data cleaning and preparation

        Args:
            spark_df (DataFrame): input spark data

        Returns:
            pd.DataFrame: spark data converted to a pandas DataFrame
        """
        logger.info("🧹 Preparing training data from Spark DataFrame...")
        df = spark_df.coalesce(1).toPandas()

        df[self.risk_colname] = (
            df[self.target_colname] > self.target_threshold
        ).astype(int)

        self._check_target_labels_proportions(df)

        logger.info("✅ Data preparation complete.")
        return df

    def prepare_inference_data(self, df: DataFrame, save: bool = True) -> DataFrame:
        """Cast data columns' types to the expected right data types

        Args:
            df (DataFrame): Input PySpark DataFrame.
            save (bool, optional): save the prepared dataframe. Defaults to True.

        Raises:
            ValueError: Unsupported type '{col_type_str}' for column '{col_name}'

        Returns:
            DataFrame: PySpark DataFrame with casted columns.
        """
        logger.info("🔄 Casting column data types...")

        type_map = {
            "integer": IntegerType(),
            "double": DoubleType(),
            "float": FloatType(),
            "string": StringType(),
        }

        for feature in self.feature_schema:
            col_name = feature["name"]
            col_type_str = feature["type"].lower()
            spark_type = type_map.get(col_type_str)

            if spark_type is None:
                logger.error("❌ Unsupported column type: '%s' for column '%s'", col_type_str, col_name)
                raise ValueError(
                    f"Unsupported type '{col_type_str}' for column '{col_name}'"
                )

            logger.debug("Casting column '%s' to type '%s'", col_name, col_type_str)
            df = df.withColumn(col_name, df[col_name].cast(spark_type))

        if save:
            logger.info("💾 Saving casted DataFrame to Delta table: %s", self.prepared_inference_data)
            df.write.format("delta").mode("overwrite").option(
                "mergeSchema", "true"
            ).saveAsTable(self.prepared_inference_data)

        logger.info("✅ Column casting complete.")
        return df
