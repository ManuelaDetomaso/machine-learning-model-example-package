import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

from diabete_prediction.prepare_data import DataPreparator


# -------------------- Fixtures --------------------

@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder.appName("test")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


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
                {"name": "BMI", "type": "double"}
            ],
        },
        "OutputData": {
            "prepared_training_data_table_name": "mock_training_table",
            "prepared_inference_data_table_name": "mock_inference_table",
        },
    }

# -------------------- Tests --------------------

def test_prepare_training_data_correct_risk_proportions(spark, mock_config):
    pdf = pd.DataFrame({"target": [50, 150]})
    sdf = spark.createDataFrame(pdf)

    expected_df = pd.DataFrame({
        "target": [50, 150],
        "Risk": [0, 1]
    })

    with patch("diabete_prediction.prepare_data.load_config", return_value=mock_config):
        preparator = DataPreparator()
        with patch("pyspark.sql.DataFrame.toPandas", return_value=expected_df):
            result_df = preparator.prepare_training_data(sdf, save=False)
            assert result_df["Risk"].tolist() == [0, 1]


def test_prepare_training_data_incorrect_risk_proportions_warns(spark, mock_config):
    mock_config["InputData"]["expected_target_proportions"] = {0: 0.2, 1: 0.8}
    pdf = pd.DataFrame({"target": [50, 150]})
    sdf = spark.createDataFrame(pdf)

    df_with_risk = pd.DataFrame({
        "target": [50, 150],
        "Risk": [0, 1]
    })

    with patch("diabete_prediction.prepare_data.load_config", return_value=mock_config):
        preparator = DataPreparator()
        with patch("pyspark.sql.DataFrame.toPandas", return_value=df_with_risk):
            with patch("warnings.warn") as warn_mock:
                preparator.prepare_training_data(sdf, save=False)
                warn_mock.assert_called_once()


def test_cast_columns_data_types(spark, mock_config):
    schema = StructType([
        StructField("AGE", IntegerType(), True),
        StructField("BMI", DoubleType(), True),
    ])
    sdf = spark.createDataFrame([(25, 22.5)], schema=schema)

    with patch("diabete_prediction.prepare_data.load_config", return_value=mock_config):
        preparator = DataPreparator()
        with patch("pyspark.sql.DataFrame.write", new_callable=MagicMock):
            result_df = preparator.prepare_inference_data(sdf, save=True)
            assert result_df.schema["AGE"].dataType == IntegerType()
            assert result_df.schema["BMI"].dataType == DoubleType()


def test_cast_columns_data_types_invalid_type_raises(spark, mock_config):
    mock_config["InputData"]["feature_schema"] = [{"name": "AGE", "type": "boolean"}]
    pdf = pd.DataFrame({"AGE": [1]})
    sdf = spark.createDataFrame(pdf)

    with patch("diabete_prediction.prepare_data.load_config", return_value=mock_config):
        preparator = DataPreparator()
        with pytest.raises(ValueError, match="Unsupported type 'boolean' for column 'AGE'"):
            preparator.prepare_inference_data(sdf, save=False)
