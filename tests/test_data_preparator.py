import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

from diabete_prediction.prepare_data import DataPreparator


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .appName("test") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    return spark

@pytest.fixture
def mock_config():
    return {
        "InputData": {
            "risk_colname": "Risk",
            "target_colname": "target",
            "target_threshold": 100,
            "expected_target_proportions": {0: 0.5, 1: 0.5},
            "feature_schema": [
                {"name": "AGE", "type": "integer"},
                {"name": "BMI", "type": "double"},
            ],
        },
        "OutputData": {
            "prepared_inference_data_table_name": "mock_table"
        },
    }


def test_prepare_training_data_correct_risk_proportions(spark, mock_config):
    pdf = pd.DataFrame({"target": [50, 150]})
    sdf = spark.createDataFrame(pdf).coalesce(1)

    with patch("diabete_prediction.prepare_data.load_config", return_value=mock_config):
        preparator = DataPreparator()

        # Silence warning output
        with patch("warnings.warn") as warn_mock:
            result_df = preparator.prepare_training_data(sdf)
            assert "Risk" in result_df.columns
            assert result_df["Risk"].tolist() == [0, 1]
            warn_mock.assert_not_called()

def test_prepare_training_data_incorrect_risk_proportions_warns(spark, mock_config):
    mock_config["InputData"]["expected_target_proportions"] = {0: 0.2, 1: 0.8}

    pdf = pd.DataFrame({"target": [50, 150]})
    sdf = spark.createDataFrame(pdf)

    with patch("diabete_prediction.condig_loader.load_config", return_value=mock_config):
        preparator = DataPreparator()

        with patch("warnings.warn") as warn_mock:
            _ = preparator.prepare_training_data(sdf)
            warn_mock.assert_called_once_with("Target lables proportions deviates from expectations.")


def test_cast_columns_data_types(spark, mock_config):
    schema = StructType([
        StructField("AGE", IntegerType(), True),
        StructField("BMI", DoubleType(), True),
    ])
    sdf = spark.createDataFrame([(25, 22.5)], schema=schema)

    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config):
        preparator = DataPreparator()

        # Mock the saveAsTable chain
        writer_mock = MagicMock()
        sdf.write.format.return_value.mode.return_value.option.return_value.saveAsTable = writer_mock

        result_df = preparator.cast_columns_data_types(sdf, save=False)
        assert result_df.schema["AGE"].dataType == IntegerType()
        assert result_df.schema["BMI"].dataType == DoubleType()
        writer_mock.assert_not_called()


def test_cast_columns_data_types_invalid_type_raises(spark, mock_config):
    mock_config["InputData"]["feature_schema"] = [{"name": "AGE", "type": "boolean"}]  # unsupported

    pdf = pd.DataFrame({"AGE": [1]})
    sdf = spark.createDataFrame(pdf)

    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config):
        preparator = DataPreparator()
        with pytest.raises(ValueError, match="Unsupported type 'boolean' for column 'AGE'"):
            preparator.cast_columns_data_types(sdf, save=False)
