import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql.functions import lit
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
        },
    }


def test_get_predict_udf_returns_udf(mock_config):
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([0.5, 0.6])

    with (
        patch("diabete_prediction.score_data.load_config", return_value=mock_config),
        patch("diabete_prediction.score_data.mlflow.pyfunc.load_model", return_value=mock_model),
    ):
        scorer = ModelScorer()
        predict_udf = scorer._get_predict_udf("models:/mock/1")
        assert callable(predict_udf)


def test_generate_predictions_dataframe(spark, mock_config):
    schema = StructType([
        StructField("AGE", IntegerType(), True),
        StructField("BMI", DoubleType(), True),
    ])
    spark_df = spark.createDataFrame([(30, 22.5), (40, 24.5)], schema=schema)

    with (
        patch("diabete_prediction.score_data.load_config", return_value=mock_config),
        patch.object(ModelScorer, "_get_predict_udf", return_value=lambda *cols: lit(1.0)),
    ):
        scorer = ModelScorer()
        df_out = scorer.generate_predictions_dataframe(
            df_inference=spark_df,
            experiment_name="ref-diabete",
            model_type="regression",
            model_version=1,
            save=False,
        )
        assert "predictions" in df_out.columns


def test_generate_predictions_dataframe_saves_when_flag_true(spark, mock_config):
    schema = StructType([
        StructField("AGE", IntegerType(), True),
        StructField("BMI", DoubleType(), True),
    ])
    spark_df = spark.createDataFrame([(25, 21.5)], schema=schema)

    with (
        patch("diabete_prediction.score_data.load_config", return_value=mock_config),
        patch.object(ModelScorer, "_get_predict_udf", return_value=lambda *cols: lit(0.9)),
        patch("pyspark.sql.DataFrame.write") as mock_write,
    ):
        mock_writer = MagicMock()
        mock_write.format.return_value.mode.return_value.option.return_value.saveAsTable = mock_writer

        scorer = ModelScorer()
        scorer.generate_predictions_dataframe(
            df_inference=spark_df,
            experiment_name="ref-diabete",
            model_type="regression",
            model_version=1,
            save=True,
        )

        mock_writer.assert_called_once_with("test_predictions_table")
