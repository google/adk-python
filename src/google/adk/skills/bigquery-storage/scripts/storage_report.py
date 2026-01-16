"""Generate BigQuery storage usage report.

This script provides detailed storage analysis including
table sizes, time travel usage, and cost estimates.

Usage:
  python storage_report.py --project PROJECT
  python storage_report.py --project PROJECT --dataset DATASET
  python storage_report.py --project PROJECT --format json
"""

import argparse
from datetime import datetime
import json
import sys


def get_storage_report(project: str, dataset: str = None) -> dict:
  """Generate storage usage report."""
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)

    # Build query
    if dataset:
      query = f"""
        SELECT
          table_schema AS dataset_id,
          table_name,
          table_type,
          total_rows,
          total_logical_bytes,
          total_physical_bytes,
          time_travel_physical_bytes,
          creation_time,
          last_modified_time
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLE_STORAGE`
        ORDER BY total_physical_bytes DESC
      """
    else:
      query = f"""
        SELECT
          table_catalog AS project_id,
          table_schema AS dataset_id,
          table_name,
          total_rows,
          total_logical_bytes,
          total_physical_bytes,
          time_travel_physical_bytes
        FROM `{project}.region-us.INFORMATION_SCHEMA.TABLE_STORAGE`
        ORDER BY total_physical_bytes DESC
        LIMIT 100
      """

    results = client.query(query).result()

    tables = []
    total_logical = 0
    total_physical = 0
    total_time_travel = 0

    for row in results:
      table_info = {
          "dataset": row.dataset_id if hasattr(row, "dataset_id") else "",
          "table": row.table_name,
          "rows": row.total_rows,
          "logical_bytes": row.total_logical_bytes,
          "physical_bytes": row.total_physical_bytes,
          "time_travel_bytes": row.time_travel_physical_bytes or 0,
      }

      # Calculate compression ratio
      if row.total_logical_bytes and row.total_physical_bytes:
        table_info["compression_ratio"] = round(
            row.total_logical_bytes / row.total_physical_bytes, 2
        )

      tables.append(table_info)
      total_logical += row.total_logical_bytes or 0
      total_physical += row.total_physical_bytes or 0
      total_time_travel += row.time_travel_physical_bytes or 0

    # Calculate costs (approximate)
    # Active storage: $0.02/GB/month, Long-term: $0.01/GB/month
    active_cost = (total_physical / 1024**3) * 0.02
    time_travel_cost = (total_time_travel / 1024**3) * 0.02

    return {
        "project": project,
        "dataset": dataset,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "table_count": len(tables),
            "total_logical_gb": round(total_logical / 1024**3, 4),
            "total_physical_gb": round(total_physical / 1024**3, 4),
            "time_travel_gb": round(total_time_travel / 1024**3, 4),
            "estimated_monthly_cost_usd": round(
                active_cost + time_travel_cost, 2
            ),
        },
        "tables": tables[:50],  # Limit to top 50
    }
  except Exception as e:
    return {"error": str(e)}


def format_bytes(bytes_val: int) -> str:
  """Format bytes to human-readable string."""
  for unit in ["B", "KB", "MB", "GB", "TB"]:
    if bytes_val < 1024:
      return f"{bytes_val:.2f} {unit}"
    bytes_val /= 1024
  return f"{bytes_val:.2f} PB"


def format_report(report: dict) -> str:
  """Format report for display."""
  output = []
  output.append("=" * 70)
  output.append("BIGQUERY STORAGE REPORT")
  output.append("=" * 70)

  if "error" in report:
    output.append(f"\nError: {report['error']}")
    return "\n".join(output)

  output.append(f"\nProject: {report['project']}")
  if report.get("dataset"):
    output.append(f"Dataset: {report['dataset']}")
  output.append(f"Generated: {report['generated_at']}")

  summary = report["summary"]
  output.append("\n## Summary")
  output.append(f"  Tables: {summary['table_count']}")
  output.append(f"  Logical Size: {summary['total_logical_gb']:.4f} GB")
  output.append(f"  Physical Size: {summary['total_physical_gb']:.4f} GB")
  output.append(f"  Time Travel: {summary['time_travel_gb']:.4f} GB")
  output.append(
      f"  Est. Monthly Cost: ${summary['estimated_monthly_cost_usd']:.2f}"
  )

  output.append("\n## Top Tables by Physical Size")
  output.append("-" * 70)
  output.append(f"{'Dataset':<20} {'Table':<25} {'Physical':<12} {'Ratio':<8}")
  output.append("-" * 70)

  for table in report["tables"][:20]:
    physical = format_bytes(table["physical_bytes"])
    ratio = table.get("compression_ratio", "-")
    output.append(
        f"{table['dataset'][:20]:<20} "
        f"{table['table'][:25]:<25} "
        f"{physical:<12} "
        f"{ratio}"
    )

  # Time travel analysis
  high_tt_tables = [
      t for t in report["tables"] if t["time_travel_bytes"] > 1024**3  # > 1GB
  ]
  if high_tt_tables:
    output.append("\n## Tables with High Time Travel Usage (>1GB)")
    output.append("-" * 70)
    for table in high_tt_tables[:10]:
      tt_size = format_bytes(table["time_travel_bytes"])
      output.append(f"  {table['dataset']}.{table['table']}: {tt_size}")

  # Recommendations
  output.append("\n## Recommendations")

  if summary["time_travel_gb"] > summary["total_physical_gb"] * 0.2:
    output.append(
        "  - Time travel storage is significant. Consider reducing "
        "max_time_travel_hours on tables with frequent updates."
    )

  low_compression = [
      t
      for t in report["tables"]
      if t.get("compression_ratio", 0) < 1.5 and t["physical_bytes"] > 1024**3
  ]
  if low_compression:
    output.append(
        "  - Some large tables have low compression. "
        "Consider using columnar formats for better compression."
    )

  output.append("\n" + "=" * 70)
  return "\n".join(output)


def main():
  parser = argparse.ArgumentParser(
      description="Generate BigQuery storage usage report"
  )
  parser.add_argument("--project", required=True, help="GCP project ID")
  parser.add_argument("--dataset", help="Specific dataset to analyze")
  parser.add_argument(
      "--format", choices=["text", "json"], default="text", help="Output format"
  )

  args = parser.parse_args()

  report = get_storage_report(args.project, args.dataset)

  if args.format == "json":
    print(json.dumps(report, indent=2, default=str))
  else:
    print(format_report(report))


if __name__ == "__main__":
  main()
