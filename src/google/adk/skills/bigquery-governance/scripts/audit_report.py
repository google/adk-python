"""Generate BigQuery access audit report.

This script analyzes BigQuery access patterns and generates
reports for security and compliance purposes.

Usage:
  python audit_report.py --project PROJECT
  python audit_report.py --project PROJECT --days 7 --format json
"""

import argparse
from datetime import datetime
import json
import sys


def get_audit_report(project: str, days: int = 7) -> dict:
  """Generate access audit report."""
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)

    # Query access patterns
    access_query = f"""
      SELECT
        user_email,
        statement_type,
        COUNT(*) AS operation_count,
        COUNT(DISTINCT DATE(creation_time)) AS active_days,
        ARRAY_AGG(DISTINCT t.dataset_id IGNORE NULLS) AS datasets_accessed,
        MIN(creation_time) AS first_access,
        MAX(creation_time) AS last_access
      FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT` j,
      UNNEST(referenced_tables) AS t
      WHERE creation_time > TIMESTAMP_SUB(
        CURRENT_TIMESTAMP(),
        INTERVAL {days} DAY
      )
      GROUP BY user_email, statement_type
      ORDER BY operation_count DESC
    """

    access_results = list(client.query(access_query).result())

    # Query data modifications
    modification_query = f"""
      SELECT
        user_email,
        destination_table.dataset_id AS dataset,
        destination_table.table_id AS table_name,
        statement_type,
        COUNT(*) AS modification_count,
        MAX(creation_time) AS last_modified
      FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
      WHERE creation_time > TIMESTAMP_SUB(
        CURRENT_TIMESTAMP(),
        INTERVAL {days} DAY
      )
      AND statement_type IN ('INSERT', 'UPDATE', 'DELETE', 'MERGE', 'TRUNCATE_TABLE')
      GROUP BY 1, 2, 3, 4
      ORDER BY modification_count DESC
      LIMIT 50
    """

    modification_results = list(client.query(modification_query).result())

    # Query failed operations
    failure_query = f"""
      SELECT
        user_email,
        error_result.reason AS error_reason,
        error_result.message AS error_message,
        COUNT(*) AS failure_count
      FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
      WHERE creation_time > TIMESTAMP_SUB(
        CURRENT_TIMESTAMP(),
        INTERVAL {days} DAY
      )
      AND error_result IS NOT NULL
      GROUP BY 1, 2, 3
      ORDER BY failure_count DESC
      LIMIT 20
    """

    failure_results = list(client.query(failure_query).result())

    # Build report
    user_activity = []
    for row in access_results:
      user_activity.append({
          "user": row.user_email,
          "operation_type": row.statement_type,
          "count": row.operation_count,
          "active_days": row.active_days,
          "datasets": (
              list(row.datasets_accessed) if row.datasets_accessed else []
          ),
          "first_access": str(row.first_access) if row.first_access else None,
          "last_access": str(row.last_access) if row.last_access else None,
      })

    modifications = []
    for row in modification_results:
      modifications.append({
          "user": row.user_email,
          "dataset": row.dataset,
          "table": row.table_name,
          "operation": row.statement_type,
          "count": row.modification_count,
          "last_modified": (
              str(row.last_modified) if row.last_modified else None
          ),
      })

    failures = []
    for row in failure_results:
      failures.append({
          "user": row.user_email,
          "reason": row.error_reason,
          "message": row.error_message[:100] if row.error_message else "",
          "count": row.failure_count,
      })

    # Calculate summary
    unique_users = len(set(r["user"] for r in user_activity))
    total_operations = sum(r["count"] for r in user_activity)
    total_modifications = sum(r["count"] for r in modifications)
    total_failures = sum(r["count"] for r in failures)

    return {
        "project": project,
        "period_days": days,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "unique_users": unique_users,
            "total_operations": total_operations,
            "total_modifications": total_modifications,
            "total_failures": total_failures,
        },
        "user_activity": user_activity[:30],
        "data_modifications": modifications,
        "access_failures": failures,
    }
  except Exception as e:
    return {"error": str(e)}


def check_anomalies(report: dict) -> list:
  """Check for anomalous access patterns."""
  anomalies = []

  # Check for unusual access times (would need timestamp analysis)
  # Check for high failure rates
  failures = report.get("access_failures", [])
  for failure in failures:
    if failure["count"] > 10 and "PERMISSION_DENIED" in (
        failure.get("reason") or ""
    ):
      anomalies.append({
          "severity": "HIGH",
          "type": "Permission Denied",
          "description": (
              f"User {failure['user']} had {failure['count']} "
              "permission denied errors. Possible unauthorized access attempt."
          ),
      })

  # Check for bulk modifications
  modifications = report.get("data_modifications", [])
  for mod in modifications:
    if mod["count"] > 100 and mod["operation"] in ("DELETE", "TRUNCATE_TABLE"):
      anomalies.append({
          "severity": "MEDIUM",
          "type": "Bulk Deletion",
          "description": (
              f"User {mod['user']} performed {mod['count']} {mod['operation']} "
              f"operations on {mod['dataset']}.{mod['table']}."
          ),
      })

  # Check for new users accessing sensitive datasets
  # (This would require a list of sensitive datasets)

  return anomalies


def format_report(report: dict, anomalies: list) -> str:
  """Format report for display."""
  output = []
  output.append("=" * 70)
  output.append("BIGQUERY ACCESS AUDIT REPORT")
  output.append("=" * 70)

  if "error" in report:
    output.append(f"\nError: {report['error']}")
    return "\n".join(output)

  output.append(f"\nProject: {report['project']}")
  output.append(f"Period: Last {report['period_days']} days")
  output.append(f"Generated: {report['generated_at']}")

  summary = report["summary"]
  output.append("\n## Summary")
  output.append(f"  Unique Users: {summary['unique_users']}")
  output.append(f"  Total Operations: {summary['total_operations']:,}")
  output.append(f"  Data Modifications: {summary['total_modifications']:,}")
  output.append(f"  Failed Operations: {summary['total_failures']:,}")

  if anomalies:
    output.append("\n## ANOMALIES DETECTED")
    output.append("-" * 70)
    for anomaly in anomalies:
      output.append(f"\n[{anomaly['severity']}] {anomaly['type']}")
      output.append(f"  {anomaly['description']}")

  output.append("\n## User Activity (Top 20)")
  output.append("-" * 70)
  output.append(f"{'User':<40} {'Operation':<15} {'Count':>10}")
  output.append("-" * 70)
  for activity in report["user_activity"][:20]:
    output.append(
        f"{activity['user'][:40]:<40} "
        f"{activity['operation_type']:<15} "
        f"{activity['count']:>10,}"
    )

  if report["data_modifications"]:
    output.append("\n## Data Modifications")
    output.append("-" * 70)
    output.append(f"{'User':<30} {'Table':<30} {'Op':<10} {'Count':>8}")
    output.append("-" * 70)
    for mod in report["data_modifications"][:15]:
      table_name = f"{mod['dataset']}.{mod['table']}"[:30]
      output.append(
          f"{mod['user'][:30]:<30} "
          f"{table_name:<30} "
          f"{mod['operation']:<10} "
          f"{mod['count']:>8,}"
      )

  if report["access_failures"]:
    output.append("\n## Access Failures")
    output.append("-" * 70)
    for failure in report["access_failures"][:10]:
      output.append(f"\nUser: {failure['user']}")
      output.append(f"  Reason: {failure['reason']}")
      output.append(f"  Count: {failure['count']}")

  output.append("\n" + "=" * 70)
  return "\n".join(output)


def main():
  parser = argparse.ArgumentParser(
      description="Generate BigQuery access audit report"
  )
  parser.add_argument("--project", required=True, help="GCP project ID")
  parser.add_argument(
      "--days",
      type=int,
      default=7,
      help="Number of days to analyze (default: 7)",
  )
  parser.add_argument(
      "--format", choices=["text", "json"], default="text", help="Output format"
  )

  args = parser.parse_args()

  report = get_audit_report(args.project, args.days)
  anomalies = check_anomalies(report)

  if args.format == "json":
    output = {"report": report, "anomalies": anomalies}
    print(json.dumps(output, indent=2, default=str))
  else:
    print(format_report(report, anomalies))


if __name__ == "__main__":
  main()
