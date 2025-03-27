import pandas as pd
from pyspark.sql import DataFrame
import warnings

from diabete_prediction.config_loader import load_config


class DataPreparator:
    def __init__(self):

        self.config = load_config()
        self.risk_colname = self.config["InputData"]["risk_colname"]
        self.target_colname = self.config["InputData"]["target_colname"]
        self.target_threshold = self.config["InputData"]["target_threshold"]
        self.expected_target_proportions = self.config["InputData"][
            "expected_target_proportions"
        ]

    def prepare_data(self, spark_df: DataFrame) -> pd.DataFrame:
        """Data cleaning and preparation

        Args:
            spark_df (DataFrame): input spark data

        Returns:
            pd.DataFrame: spark data converted to a pandas DataFrame
        """
        df = spark_df.toPandas()
        # Created column 'Risk' from formula
        df[self.risk_colname] = (
            df[self.target_colname] > self.target_threshold
        ).astype(int)
        # check that the Risk column labels' proportion is as expected
        self._check_target_labels_proportions(df)

        return df

    def _check_target_labels_proportions(self, df: pd.DataFrame):
        """Checks whether the categorised target values assume expected proportions

        Args:
            df (pd.DataFrame): prepared data
        """
        proportion_dict = dict(
            round(df[self.risk_colname].value_counts() / len(df[self.risk_colname]), 2)
        )
        if proportion_dict != self.expected_target_proportions:
            warnings.warn("Target lables proportions deviates from expectations.")
        else:
            print("Target values proprotions as expected.")
