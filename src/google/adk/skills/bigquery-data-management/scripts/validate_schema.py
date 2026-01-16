"""Validate data files against BigQuery table schema.

This script checks if data files (CSV, JSON, Parquet) are compatible
with a BigQuery table schema before loading.

Usage:
  python validate_schema.py --project PROJECT --dataset DATASET --table TABLE --file FILE
  python validate_schema.py --schema schema.json --file data.csv
"""

import argparse
import json
from pathlib import Path
import sys


def get_bigquery_schema(project: str, dataset: str, table: str) -> list:
  """Fetch schema from BigQuery table."""
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    table_ref = client.get_table(f"{project}.{dataset}.{table}")
    return [
        {"name": f.name, "type": f.field_type, "mode": f.mode}
        for f in table_ref.schema
    ]
  except Exception as e:
    print(f"Error fetching schema: {e}", file=sys.stderr)
    sys.exit(1)


def load_schema_file(schema_path: str) -> list:
  """Load schema from JSON file."""
  with open(schema_path) as f:
    return json.load(f)


def infer_csv_schema(file_path: str, delimiter: str = ",") -> list:
  """Infer schema from CSV file header."""
  import csv

  with open(file_path, newline="") as f:
    reader = csv.reader(f, delimiter=delimiter)
    header = next(reader)
    first_row = next(reader, None)

    schema = []
    for i, col in enumerate(header):
      col_type = "STRING"
      if first_row:
        val = first_row[i] if i < len(first_row) else ""
        col_type = infer_type(val)
      schema.append({"name": col, "type": col_type, "mode": "NULLABLE"})
    return schema


def infer_json_schema(file_path: str) -> list:
  """Infer schema from JSON file."""
  with open(file_path) as f:
    # Read first line for newline-delimited JSON
    line = f.readline()
    obj = json.loads(line)

    schema = []
    for key, value in obj.items():
      col_type = infer_type_from_value(value)
      schema.append({"name": key, "type": col_type, "mode": "NULLABLE"})
    return schema


def infer_parquet_schema(file_path: str) -> list:
  """Infer schema from Parquet file."""
  try:
    import pyarrow.parquet as pq

    table = pq.read_table(file_path)
    schema = []
    for field in table.schema:
      bq_type = arrow_to_bigquery_type(str(field.type))
      schema.append({
          "name": field.name,
          "type": bq_type,
          "mode": "NULLABLE" if field.nullable else "REQUIRED",
      })
    return schema
  except ImportError:
    print("pyarrow required for Parquet validation", file=sys.stderr)
    sys.exit(1)


def arrow_to_bigquery_type(arrow_type: str) -> str:
  """Convert Arrow type to BigQuery type."""
  type_map = {
      "int64": "INT64",
      "int32": "INT64",
      "float64": "FLOAT64",
      "float32": "FLOAT64",
      "double": "FLOAT64",
      "bool": "BOOL",
      "string": "STRING",
      "binary": "BYTES",
      "date32": "DATE",
      "timestamp": "TIMESTAMP",
  }
  for key, val in type_map.items():
    if key in arrow_type.lower():
      return val
  return "STRING"


def infer_type(value: str) -> str:
  """Infer BigQuery type from string value."""
  if not value:
    return "STRING"
  try:
    int(value)
    return "INT64"
  except ValueError:
    pass
  try:
    float(value)
    return "FLOAT64"
  except ValueError:
    pass
  if value.lower() in ("true", "false"):
    return "BOOL"
  return "STRING"


def infer_type_from_value(value) -> str:
  """Infer BigQuery type from Python value."""
  if isinstance(value, bool):
    return "BOOL"
  if isinstance(value, int):
    return "INT64"
  if isinstance(value, float):
    return "FLOAT64"
  if isinstance(value, list):
    return "ARRAY"
  if isinstance(value, dict):
    return "STRUCT"
  return "STRING"


def validate_schema(expected: list, actual: list) -> list:
  """Compare schemas and return list of issues."""
  issues = []
  expected_map = {f["name"].lower(): f for f in expected}
  actual_map = {f["name"].lower(): f for f in actual}

  # Check for missing columns
  for name, field in expected_map.items():
    if name not in actual_map:
      if field.get("mode") == "REQUIRED":
        issues.append(f"MISSING REQUIRED: Column '{name}' not found in data")
      else:
        issues.append(f"WARNING: Column '{name}' not found in data")

  # Check for extra columns
  for name in actual_map:
    if name not in expected_map:
      issues.append(f"EXTRA: Column '{name}' in data not in schema")

  # Check type compatibility
  for name, expected_field in expected_map.items():
    if name in actual_map:
      actual_field = actual_map[name]
      if not types_compatible(expected_field["type"], actual_field["type"]):
        issues.append(
            f"TYPE MISMATCH: Column '{name}' expected "
            f"{expected_field['type']}, got {actual_field['type']}"
        )

  return issues


def types_compatible(expected: str, actual: str) -> bool:
  """Check if types are compatible for loading."""
  expected = expected.upper()
  actual = actual.upper()

  if expected == actual:
    return True

  # String can accept anything
  if expected == "STRING":
    return True

  # Numeric compatibility
  numeric_types = {"INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC"}
  if expected in numeric_types and actual in numeric_types:
    return True

  return False


def main():
  parser = argparse.ArgumentParser(
      description="Validate data files against BigQuery schema"
  )
  parser.add_argument("--project", help="GCP project ID")
  parser.add_argument("--dataset", help="BigQuery dataset")
  parser.add_argument("--table", help="BigQuery table")
  parser.add_argument("--schema", help="Path to schema JSON file")
  parser.add_argument("--file", required=True, help="Data file to validate")
  parser.add_argument("--delimiter", default=",", help="CSV delimiter")

  args = parser.parse_args()

  # Get expected schema
  if args.schema:
    expected_schema = load_schema_file(args.schema)
  elif args.project and args.dataset and args.table:
    expected_schema = get_bigquery_schema(
        args.project, args.dataset, args.table
    )
  else:
    print("Provide either --schema or --project/--dataset/--table")
    sys.exit(1)

  # Infer actual schema from file
  file_path = Path(args.file)
  if file_path.suffix.lower() == ".csv":
    actual_schema = infer_csv_schema(args.file, args.delimiter)
  elif file_path.suffix.lower() in (".json", ".jsonl"):
    actual_schema = infer_json_schema(args.file)
  elif file_path.suffix.lower() == ".parquet":
    actual_schema = infer_parquet_schema(args.file)
  else:
    print(f"Unsupported file format: {file_path.suffix}")
    sys.exit(1)

  # Validate
  issues = validate_schema(expected_schema, actual_schema)

  if issues:
    print("Schema validation issues found:")
    for issue in issues:
      print(f"  - {issue}")
    sys.exit(1)
  else:
    print("Schema validation passed")
    sys.exit(0)


if __name__ == "__main__":
  main()
