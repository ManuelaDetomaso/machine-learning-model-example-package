import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession

from diabete_prediction.train_model import ModelTrainer
from unittest.mock import patch, MagicMock
from mlflow.types.schema import Schema, ColSpec
import pandas as pd


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .appName("test") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def mock_config():
    return {
        "InputData": {
            "risk_colname": "Risk",
            "target_colname": "target",
            "feature_schema": [
                {"name": "AGE", "type": "integer"},
                {"name": "BMI", "type": "double"},
            ],
        },
        "ModelTraining": {
            "test_size": 0.2,
        },
        "ModelGeneral": {
            "experiment_name": "test-experiment"
        }
    }


def test_split_train_test_data_numerical(spark, mock_config):
    data = pd.DataFrame({"AGE": [20, 30, 40, 50, 60], "BMI": [22, 23, 24, 25, 26], "target": [1, 2, 3, 4, 5]})
    spark_df = spark.createDataFrame(data).coalesce(1)
    

    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config):
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_train_test_data(spark_df, target_type="numerical")

        assert X_train.shape[1] == 2
        assert len(y_train) + len(y_test) == 5


def test_split_train_test_data_invalid_type_raises(spark, mock_config):
    data = pd.DataFrame({"AGE": [20], "BMI": [22], "target": [1]})
    spark_df = spark.createDataFrame(data)

    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config):
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Indicate a target type, either numerical or categorical"):
            trainer.split_train_test_data(spark_df, target_type="unknown")


@patch("diabete_prediction.train_model.create_mlflow_schema_from_typed_list")
@patch("diabete_prediction.train_model.mlflow")
def test_mlflow_training_logs_model(mock_mlflow, mock_schema_func):
    # ✅ Set the return value of the schema patch here
    mock_schema_func.return_value = Schema([
        ColSpec("long", "AGE"),
        ColSpec("double", "BMI")
    ])

    # Optional: mock the MLflow log_model and autolog to avoid actual logging
    mock_mlflow.sklearn.log_model = MagicMock()
    mock_mlflow.autolog = MagicMock()
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Example input data
    X = pd.DataFrame([[25, 22.5], [30, 24.0]], columns=["AGE", "BMI"])
    y = pd.Series([0, 1])

    from diabete_prediction.train_model import ModelTrainer
    trainer = ModelTrainer()

    with patch("sklearn.linear_model.LinearRegression.fit", return_value=None):
        trainer.mlflow_training(
            experiment_name="test-experiment",
            X_train=X,
            y_train=y,
            model_type="regression",
            logs=False,
            save_model=False
        )


@patch("diabete_prediction.train_model.mlflow")
def test_save_mlflow_model_registers_model(mock_mlflow, mock_config):
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp123"

    mock_mlflow.get_experiment_by_name.return_value = mock_experiment
    mock_mlflow.search_runs.return_value = pd.DataFrame([{"run_id": "run_456"}])
    mock_mlflow.register_model.return_value = MagicMock(name="test", version="1")

    with patch("diabete_prediction.config_loader.load_config", return_value=mock_config):
        trainer = ModelTrainer()
        trainer.save_mlflow_model("test-experiment")

        mock_mlflow.get_experiment_by_name.assert_called_with("test-experiment")
        mock_mlflow.search_runs.assert_called()
        mock_mlflow.register_model.assert_called_with("runs:/run_456/model", "test-experiment-model")