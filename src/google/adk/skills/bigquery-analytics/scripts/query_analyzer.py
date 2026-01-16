"""Analyze BigQuery query performance and suggest optimizations.

This script helps analyze query execution plans and provides
recommendations for improving query performance.

Usage:
  python query_analyzer.py --project PROJECT --query "SELECT ..."
  python query_analyzer.py --project PROJECT --job-id JOB_ID
"""

import argparse
import json
import sys
from typing import Any


def analyze_dry_run(project: str, query: str) -> dict:
  """Perform dry run analysis of a query."""
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    query_job = client.query(query, job_config=job_config)

    return {
        "total_bytes_processed": query_job.total_bytes_processed,
        "total_bytes_billed": query_job.total_bytes_billed,
        "estimated_cost_usd": query_job.total_bytes_billed / (1024**4) * 5,
        "referenced_tables": [
            f"{t.project}.{t.dataset_id}.{t.table_id}"
            for t in (query_job.referenced_tables or [])
        ],
        "schema": [
            {"name": f.name, "type": f.field_type}
            for f in (query_job.schema or [])
        ],
    }
  except Exception as e:
    return {"error": str(e)}


def analyze_job(project: str, job_id: str) -> dict:
  """Analyze a completed job's performance."""
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    job = client.get_job(job_id)

    stats = job.query_plan if hasattr(job, "query_plan") else []
    timeline = job.timeline if hasattr(job, "timeline") else []

    analysis = {
        "job_id": job_id,
        "state": job.state,
        "total_bytes_processed": job.total_bytes_processed,
        "total_bytes_billed": job.total_bytes_billed,
        "slot_millis": getattr(job, "slot_millis", None),
        "cache_hit": getattr(job, "cache_hit", False),
        "creation_time": str(job.created),
        "start_time": str(job.started) if job.started else None,
        "end_time": str(job.ended) if job.ended else None,
        "execution_ms": (
            (job.ended - job.started).total_seconds() * 1000
            if job.ended and job.started
            else None
        ),
        "stage_count": len(stats),
        "stages": [],
    }

    for stage in stats:
      stage_info = {
          "name": stage.name,
          "id": stage.id,
          "status": stage.status,
          "input_stages": (
              list(stage.input_stages) if stage.input_stages else []
          ),
          "records_read": stage.records_read,
          "records_written": stage.records_written,
          "shuffle_output_bytes": stage.shuffle_output_bytes,
          "steps": [],
      }

      for step in stage.steps or []:
        stage_info["steps"].append({
            "kind": step.kind,
            "substeps": list(step.substeps) if step.substeps else [],
        })

      analysis["stages"].append(stage_info)

    return analysis
  except Exception as e:
    return {"error": str(e)}


def suggest_optimizations(analysis: dict) -> list:
  """Generate optimization suggestions based on analysis."""
  suggestions = []

  # Check bytes processed
  bytes_processed = analysis.get("total_bytes_processed", 0)
  if bytes_processed and bytes_processed > 10 * 1024**3:  # 10 GB
    suggestions.append({
        "severity": "HIGH",
        "category": "Data Volume",
        "suggestion": (
            f"Query processes {bytes_processed / 1024**3:.2f} GB. "
            "Consider partitioning/clustering tables, or adding filters."
        ),
    })

  # Check for cache hit
  if analysis.get("cache_hit") is False:
    suggestions.append({
        "severity": "LOW",
        "category": "Caching",
        "suggestion": (
            "Query didn't hit cache. For repeated queries, ensure "
            "deterministic queries to benefit from caching."
        ),
    })

  # Check slot usage
  slot_millis = analysis.get("slot_millis")
  exec_ms = analysis.get("execution_ms")
  if slot_millis and exec_ms and exec_ms > 0:
    parallelism = slot_millis / exec_ms
    if parallelism < 10:
      suggestions.append({
          "severity": "MEDIUM",
          "category": "Parallelism",
          "suggestion": (
              f"Low parallelism detected ({parallelism:.1f}x). "
              "Query may be bottlenecked on sequential operations."
          ),
      })

  # Check stages
  stages = analysis.get("stages", [])
  for stage in stages:
    # Large shuffle
    shuffle_bytes = stage.get("shuffle_output_bytes", 0)
    if shuffle_bytes and shuffle_bytes > 1 * 1024**3:  # 1 GB
      suggestions.append({
          "severity": "MEDIUM",
          "category": "Shuffle",
          "suggestion": (
              f"Stage '{stage['name']}' has large shuffle "
              f"({shuffle_bytes / 1024**3:.2f} GB). "
              "Consider reducing data before joins/aggregations."
          ),
      })

    # Check for expensive operations
    for step in stage.get("steps", []):
      kind = step.get("kind", "")
      if kind == "CROSS_JOIN":
        suggestions.append({
            "severity": "HIGH",
            "category": "Query Pattern",
            "suggestion": (
                f"CROSS JOIN detected in stage '{stage['name']}'. "
                "This can be very expensive. Consider adding join predicates."
            ),
        })
      if kind == "SORT":
        suggestions.append({
            "severity": "LOW",
            "category": "Sorting",
            "suggestion": (
                f"Sort operation in stage '{stage['name']}'. "
                "ORDER BY on large results is expensive. "
                "Add LIMIT or sort in application."
            ),
        })

  # No suggestions
  if not suggestions:
    suggestions.append({
        "severity": "INFO",
        "category": "General",
        "suggestion": "No obvious optimization opportunities found.",
    })

  return suggestions


def format_output(analysis: dict, suggestions: list) -> str:
  """Format analysis results for display."""
  output = []
  output.append("=" * 60)
  output.append("BIGQUERY QUERY ANALYSIS")
  output.append("=" * 60)

  if "error" in analysis:
    output.append(f"\nError: {analysis['error']}")
    return "\n".join(output)

  # Summary
  output.append("\n## Summary")
  bytes_p = analysis.get("total_bytes_processed", 0)
  bytes_b = analysis.get("total_bytes_billed", 0)
  output.append(f"  Bytes Processed: {bytes_p / 1024**3:.4f} GB")
  output.append(f"  Bytes Billed: {bytes_b / 1024**3:.4f} GB")
  output.append(
      f"  Estimated Cost: ${analysis.get('estimated_cost_usd', 0):.4f}"
  )

  if analysis.get("cache_hit"):
    output.append("  Cache Hit: Yes (no bytes billed)")

  exec_ms = analysis.get("execution_ms")
  if exec_ms:
    output.append(f"  Execution Time: {exec_ms:.0f} ms")

  # Referenced tables
  tables = analysis.get("referenced_tables", [])
  if tables:
    output.append("\n## Referenced Tables")
    for t in tables:
      output.append(f"  - {t}")

  # Stages
  stages = analysis.get("stages", [])
  if stages:
    output.append(f"\n## Query Stages ({len(stages)} stages)")
    for stage in stages[:5]:  # Limit to first 5
      output.append(f"\n  Stage: {stage['name']}")
      output.append(f"    Records Read: {stage.get('records_read', 'N/A')}")
      output.append(
          f"    Records Written: {stage.get('records_written', 'N/A')}"
      )

  # Suggestions
  output.append("\n## Optimization Suggestions")
  for s in suggestions:
    output.append(f"\n  [{s['severity']}] {s['category']}")
    output.append(f"    {s['suggestion']}")

  output.append("\n" + "=" * 60)
  return "\n".join(output)


def main():
  parser = argparse.ArgumentParser(
      description="Analyze BigQuery query performance"
  )
  parser.add_argument("--project", required=True, help="GCP project ID")
  parser.add_argument("--query", help="SQL query to analyze (dry run)")
  parser.add_argument("--job-id", help="Completed job ID to analyze")
  parser.add_argument("--json", action="store_true", help="Output as JSON")

  args = parser.parse_args()

  if not args.query and not args.job_id:
    print("Provide either --query or --job-id")
    sys.exit(1)

  if args.query:
    analysis = analyze_dry_run(args.project, args.query)
  else:
    analysis = analyze_job(args.project, args.job_id)

  suggestions = suggest_optimizations(analysis)

  if args.json:
    print(
        json.dumps(
            {"analysis": analysis, "suggestions": suggestions},
            indent=2,
            default=str,
        )
    )
  else:
    print(format_output(analysis, suggestions))


if __name__ == "__main__":
  main()
