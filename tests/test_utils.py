import pytest

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
    FloatType,
)
from mlflow.types.schema import Schema, ColSpec

from diabete_prediction.utils import (
    parse_value,
    create_spark_schema_from_typed_list,
    create_mlflow_schema_from_typed_list,
)


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (" 42 ", 42),
        ("3.14", 3.14),
        ("true", True),
        ("False", False),
        ("[1, 2, 3]", [1, 2, 3]),
        ('{"a": 1}', {"a": 1}),
        ("'string'", "string"),  # fallback via literal_eval
        ("invalid json", "invalid json"),  # fallback to string
    ],
)
def test_parse_value(input_value, expected):
    assert parse_value(input_value) == expected


def test_create_spark_schema_from_typed_list():
    feature_list = [
        {"name": "age", "type": "integer"},
        {"name": "bmi", "type": "double"},
        {"name": "name", "type": "string"},
        {"name": "score", "type": "float"},
    ]
    schema = create_spark_schema_from_typed_list(feature_list)
    expected = StructType(
        [
            StructField("age", IntegerType(), True),
            StructField("bmi", DoubleType(), True),
            StructField("name", StringType(), True),
            StructField("score", FloatType(), True),
        ]
    )
    assert schema == expected


def test_create_spark_schema_raises_on_unknown_type():
    feature_list = [{"name": "x", "type": "boolean"}]  # unsupported
    with pytest.raises(ValueError, match="Unsupported field type: 'boolean'"):
        create_spark_schema_from_typed_list(feature_list)


def test_create_mlflow_schema_from_typed_list():
    feature_list = [
        {"name": "age", "type": "integer"},
        {"name": "bmi", "type": "double"},
        {"name": "name", "type": "string"},
        {"name": "score", "type": "float"},
    ]
    schema = create_mlflow_schema_from_typed_list(feature_list)
    expected = Schema(
        [
            ColSpec("long", "age"),
            ColSpec("double", "bmi"),
            ColSpec("string", "name"),
            ColSpec("float", "score"),
        ]
    )
    assert schema == expected


def test_create_mlflow_schema_raises_on_unknown_type():
    feature_list = [{"name": "flag", "type": "boolean"}]
    with pytest.raises(ValueError, match="Unsupported MLflow data type: boolean"):
        create_mlflow_schema_from_typed_list(feature_list)
