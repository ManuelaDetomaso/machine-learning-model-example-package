import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

from diabete_prediction.score_data import ModelScorer


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[*]").appName("test").getOrCreate()


@pytest.fixture
def mock_config():
    return {
        "InputData": {
            "feature_schema": [
                {"name": "AGE", "type": "integer"},
                {"name": "BMI", "type": "double"},
            ]
        },
        "OutputData": {
            "predictions_table_name": "test_predictions_table"
        }
    }


def test_get_predict_udf_returns_udf(mock_config):
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([0.5, 0.6])

    # Patch load_config and mlflow model loading
    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config), \
         patch("diabete_prediction.score_data.mlflow.pyfunc.load_model", return_value=mock_model):
        
        scorer = ModelScorer()
        predict_udf = scorer._get_predict_udf("models:/mock/1")

        # Ensure the returned object is callable (a UDF)
        assert callable(predict_udf)


def test_generate_predictions_dataframe(spark, mock_config):
    # Create test Spark DataFrame
    schema = StructType([
        StructField("AGE", IntegerType(), True),
        StructField("BMI", DoubleType(), True),
    ])
    spark_df = spark.createDataFrame([(30, 22.5), (40, 24.5)], schema=schema)

    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([0.1, 0.2])

    # Mock UDF to return a constant column (simulate withColumn behavior)
    def fake_udf(*args):
        return spark_df.withColumn("predictions", spark_df["AGE"].cast("float"))
    
    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config), \
         patch("diabete_prediction.score_data.mlflow.pyfunc.load_model", return_value=mock_model), \
         patch("diabete_prediction.score_data.pandas_udf", return_value=fake_udf), \
         patch("diabete_prediction.score_data.DataFrame.write") as mock_write:

        scorer = ModelScorer()
        df_out = scorer.generate_predictions_dataframe(
            df_inference=spark_df,
            experiment_name="ref-diabete",
            model_type="regression",
            model_version=1,
            save=False
        )

        assert "predictions" in df_out.columns
        mock_write.format.return_value.mode.return_value.option.return_value.saveAsTable.assert_not_called()


def test_generate_predictions_dataframe_saves_when_flag_true(spark, mock_config):
    # Create Spark DataFrame
    schema = StructType([
        StructField("AGE", IntegerType(), True),
        StructField("BMI", DoubleType(), True),
    ])
    spark_df = spark.createDataFrame([(20, 21.0)], schema=schema)

    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([0.3])

    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config), \
         patch("diabete_prediction.score_data.mlflow.pyfunc.load_model", return_value=mock_model), \
         patch("diabete_prediction.score_data.pandas_udf") as mock_pandas_udf:

        scorer = ModelScorer()
        
        # Fake the UDF behavior by returning a callable that adds a predictions column
        def fake_udf(*args):
            return spark_df.withColumn("predictions", spark_df["AGE"].cast("float"))
        
        mock_pandas_udf.return_value = fake_udf
        mock_writer = MagicMock()
        spark_df.write.format.return_value.mode.return_value.option.return_value.saveAsTable = mock_writer

        scorer.generate_predictions_dataframe(
            df_inference=spark_df,
            experiment_name="ref-diabete",
            model_type="regression",
            model_version=1,
            save=True
        )

        mock_writer.assert_called_once_with(mock_config["OutputData"]["predictions_table_name"])
