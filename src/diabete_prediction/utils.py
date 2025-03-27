from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
    FloatType,
)
import json
import ast


def parse_value(value):
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


def create_spark_schema_from_typed_list(feature_list):
    """Convert a list of {"name": ..., "type": ...} dictionaries into a StructType schema.
    Each field in the resulting schema will be nullable and use the appropriate Spark type.

    Raises:
        ValueError: Unsupported field type:

    Returns:
        _type_: _description_
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
