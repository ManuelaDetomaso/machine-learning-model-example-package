import ast
import json
from typing import Any, Dict, List

from mlflow.types.schema import ColSpec, Schema
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def parse_value(value: Any):
    """Attempt to convert config values to appropriate Python types."""
    value = value.strip()
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return json.loads(value)  # for lists, dicts
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)  # fallback
    except Exception:
        return value


def create_spark_schema_from_typed_list(feature_list: List[Dict[str, Any]]):
    """Convert a list of {"name": ..., "type": ...} dictionaries into a StructType schema.
    Each field in the resulting schema will be nullable and use the appropriate Spark type.

    Raises:
        ValueError: Unsupported field type:

    Returns:
        StructType: pysark StructType schema
    """
    # Mapping from string type names to PySpark data type classes
    type_map = {
        "integer": IntegerType(),
        "double": DoubleType(),
        "string": StringType(),
        "float": FloatType(),
    }
    fields = []
    for field_def in feature_list:
        field_name = field_def.get("name")
        field_type_str = field_def.get("type", "").lower()  # normalize to lowercase
        # Look up the Spark data type class from the mapping
        data_type = type_map.get(field_type_str)
        if data_type is None:
            raise ValueError(f"Unsupported field type: '{field_type_str}'")
        # Create StructField with nullable=True
        fields.append(StructField(field_name, data_type, True))
    # Create StructType schema from the list of StructField objects
    schema = StructType(fields)
    return schema


def create_mlflow_schema_from_typed_list(feature_list: List[Dict[str, Any]]) -> Schema:
    """Create an mlflow schema from typed list

    Args:
        feature_list (List[Dict[str, Any]]): list of data columns dictionaries with their names and type

    Raises:
        ValueError: Unsupported MLflow data type: {dtype}

    Returns:
        Schema: mlflow Schema
    """
    type_map = {
        "integer": "long",
        "double": "double",
        "string": "string",
        "float": "float",
    }

    cols = []
    for feature in feature_list:
        name = feature["name"]
        dtype = feature["type"].lower()
        mlflow_type = type_map.get(dtype)
        if mlflow_type is None:
            raise ValueError(f"Unsupported MLflow data type: {dtype}")
        cols.append(ColSpec(type=mlflow_type, name=name))

    return Schema(cols)
