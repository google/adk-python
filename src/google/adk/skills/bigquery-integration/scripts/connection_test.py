"""Test BigQuery connectivity and permissions.

This script verifies BigQuery connection and checks
available permissions for the authenticated user.

Usage:
  python connection_test.py --project PROJECT
  python connection_test.py --project PROJECT --dataset DATASET
"""

import argparse
from datetime import datetime
import json
import sys


def test_connection(project: str, dataset: str = None) -> dict:
  """Test BigQuery connection and permissions."""
  results = {
      "project": project,
      "timestamp": datetime.utcnow().isoformat(),
      "tests": [],
  }

  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    results["tests"].append({
        "test": "Client initialization",
        "status": "PASS",
        "message": "Successfully created BigQuery client",
    })
  except Exception as e:
    results["tests"].append(
        {"test": "Client initialization", "status": "FAIL", "message": str(e)}
    )
    return results

  # Test list datasets
  try:
    datasets = list(client.list_datasets(max_results=5))
    results["tests"].append({
        "test": "List datasets",
        "status": "PASS",
        "message": f"Found {len(datasets)} datasets",
        "details": [d.dataset_id for d in datasets],
    })
  except Exception as e:
    results["tests"].append(
        {"test": "List datasets", "status": "FAIL", "message": str(e)}
    )

  # Test simple query
  try:
    query = "SELECT 1 as test_value"
    result = list(client.query(query).result())
    if result[0].test_value == 1:
      results["tests"].append({
          "test": "Execute simple query",
          "status": "PASS",
          "message": "Successfully executed test query",
      })
    else:
      results["tests"].append({
          "test": "Execute simple query",
          "status": "FAIL",
          "message": "Query returned unexpected result",
      })
  except Exception as e:
    results["tests"].append(
        {"test": "Execute simple query", "status": "FAIL", "message": str(e)}
    )

  # Test INFORMATION_SCHEMA access
  try:
    query = """
      SELECT COUNT(*) as job_count
      FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
      WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    """
    result = list(client.query(query).result())
    results["tests"].append({
        "test": "INFORMATION_SCHEMA access",
        "status": "PASS",
        "message": f"Found {result[0].job_count} jobs in last hour",
    })
  except Exception as e:
    results["tests"].append({
        "test": "INFORMATION_SCHEMA access",
        "status": "FAIL",
        "message": str(e),
    })

  # Test specific dataset if provided
  if dataset:
    # Test dataset access
    try:
      ds = client.get_dataset(f"{project}.{dataset}")
      results["tests"].append({
          "test": f"Access dataset {dataset}",
          "status": "PASS",
          "message": f"Dataset location: {ds.location}",
      })
    except Exception as e:
      results["tests"].append({
          "test": f"Access dataset {dataset}",
          "status": "FAIL",
          "message": str(e),
      })

    # Test list tables
    try:
      tables = list(client.list_tables(f"{project}.{dataset}", max_results=10))
      results["tests"].append({
          "test": f"List tables in {dataset}",
          "status": "PASS",
          "message": f"Found {len(tables)} tables",
          "details": [t.table_id for t in tables],
      })
    except Exception as e:
      results["tests"].append({
          "test": f"List tables in {dataset}",
          "status": "FAIL",
          "message": str(e),
      })

  # Test dry run (query cost estimation)
  try:
    query = (
        f"SELECT * FROM `{project}.{dataset or 'information_schema'}.TABLES`"
        " LIMIT 100"
    )
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)
    results["tests"].append({
        "test": "Dry run query",
        "status": "PASS",
        "message": f"Estimated bytes: {job.total_bytes_processed}",
    })
  except Exception as e:
    results["tests"].append(
        {"test": "Dry run query", "status": "FAIL", "message": str(e)}
    )

  # Summary
  passed = sum(1 for t in results["tests"] if t["status"] == "PASS")
  failed = sum(1 for t in results["tests"] if t["status"] == "FAIL")
  results["summary"] = {
      "total_tests": len(results["tests"]),
      "passed": passed,
      "failed": failed,
      "status": "HEALTHY" if failed == 0 else "UNHEALTHY",
  }

  return results


def format_results(results: dict) -> str:
  """Format test results for display."""
  output = []
  output.append("=" * 60)
  output.append("BIGQUERY CONNECTION TEST")
  output.append("=" * 60)
  output.append(f"\nProject: {results['project']}")
  output.append(f"Timestamp: {results['timestamp']}")

  output.append("\n## Test Results")
  output.append("-" * 60)

  for test in results["tests"]:
    status_icon = "[PASS]" if test["status"] == "PASS" else "[FAIL]"
    output.append(f"\n{status_icon} {test['test']}")
    output.append(f"  {test['message']}")
    if "details" in test:
      for detail in test["details"][:5]:
        output.append(f"    - {detail}")
      if len(test.get("details", [])) > 5:
        output.append(f"    ... and {len(test['details']) - 5} more")

  summary = results["summary"]
  output.append("\n## Summary")
  output.append("-" * 60)
  output.append(f"  Total Tests: {summary['total_tests']}")
  output.append(f"  Passed: {summary['passed']}")
  output.append(f"  Failed: {summary['failed']}")
  output.append(f"  Status: {summary['status']}")

  output.append("\n" + "=" * 60)
  return "\n".join(output)


def main():
  parser = argparse.ArgumentParser(
      description="Test BigQuery connectivity and permissions"
  )
  parser.add_argument("--project", required=True, help="GCP project ID")
  parser.add_argument("--dataset", help="Specific dataset to test")
  parser.add_argument(
      "--format", choices=["text", "json"], default="text", help="Output format"
  )

  args = parser.parse_args()

  results = test_connection(args.project, args.dataset)

  if args.format == "json":
    print(json.dumps(results, indent=2))
  else:
    print(format_results(results))

  # Exit with error code if tests failed
  sys.exit(0 if results["summary"]["status"] == "HEALTHY" else 1)


if __name__ == "__main__":
  main()
