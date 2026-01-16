"""Generate BigQuery cost analysis report.

This script analyzes query costs and provides recommendations
for cost optimization.

Usage:
  python cost_report.py --project PROJECT
  python cost_report.py --project PROJECT --days 30 --format json
"""

import argparse
from datetime import datetime
from datetime import timedelta
import json
import sys


def get_cost_report(project: str, days: int = 30) -> dict:
  """Generate cost analysis report."""
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)

    # Query cost data from INFORMATION_SCHEMA
    query = f"""
      WITH job_costs AS (
        SELECT
          user_email,
          DATE(creation_time) AS query_date,
          job_id,
          total_bytes_billed,
          total_slot_ms,
          cache_hit,
          statement_type,
          ARRAY_TO_STRING(
            ARRAY(
              SELECT CONCAT(t.project_id, '.', t.dataset_id, '.', t.table_id)
              FROM UNNEST(referenced_tables) t
            ),
            ', '
          ) AS tables_accessed
        FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
        WHERE creation_time > TIMESTAMP_SUB(
          CURRENT_TIMESTAMP(),
          INTERVAL {days} DAY
        )
        AND statement_type = 'SELECT'
        AND total_bytes_billed > 0
      )
      SELECT
        user_email,
        COUNT(*) AS query_count,
        SUM(total_bytes_billed) AS total_bytes_billed,
        SUM(total_slot_ms) AS total_slot_ms,
        COUNTIF(cache_hit) AS cache_hits,
        COUNTIF(NOT cache_hit) AS cache_misses
      FROM job_costs
      GROUP BY user_email
      ORDER BY total_bytes_billed DESC
      LIMIT 50
    """

    results = list(client.query(query).result())

    # Calculate totals
    total_bytes = sum(r.total_bytes_billed or 0 for r in results)
    total_queries = sum(r.query_count or 0 for r in results)
    total_cache_hits = sum(r.cache_hits or 0 for r in results)

    # On-demand pricing: $5 per TB
    estimated_cost = (total_bytes / 1e12) * 5

    users = []
    for row in results:
      user_cost = (row.total_bytes_billed / 1e12) * 5
      users.append({
          "email": row.user_email,
          "query_count": row.query_count,
          "tb_scanned": round(row.total_bytes_billed / 1e12, 6),
          "slot_hours": round(row.total_slot_ms / 1000 / 60 / 60, 2),
          "cache_hit_rate": round(
              row.cache_hits / (row.cache_hits + row.cache_misses) * 100
              if (row.cache_hits + row.cache_misses) > 0
              else 0,
              1,
          ),
          "estimated_cost_usd": round(user_cost, 2),
      })

    # Get expensive queries
    expensive_query = f"""
      SELECT
        job_id,
        user_email,
        total_bytes_billed,
        TIMESTAMP_DIFF(end_time, start_time, SECOND) AS duration_sec,
        query
      FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
      WHERE creation_time > TIMESTAMP_SUB(
        CURRENT_TIMESTAMP(),
        INTERVAL {days} DAY
      )
      AND statement_type = 'SELECT'
      ORDER BY total_bytes_billed DESC
      LIMIT 10
    """

    expensive_results = list(client.query(expensive_query).result())
    expensive_queries = []
    for row in expensive_results:
      expensive_queries.append({
          "job_id": row.job_id,
          "user": row.user_email,
          "tb_scanned": round(row.total_bytes_billed / 1e12, 6),
          "duration_sec": row.duration_sec,
          "cost_usd": round((row.total_bytes_billed / 1e12) * 5, 2),
          "query_preview": (
              row.query[:200] + "..." if len(row.query) > 200 else row.query
          ),
      })

    return {
        "project": project,
        "period_days": days,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_queries": total_queries,
            "total_tb_scanned": round(total_bytes / 1e12, 4),
            "estimated_cost_usd": round(estimated_cost, 2),
            "unique_users": len(users),
            "cache_hit_rate": round(
                total_cache_hits / total_queries * 100
                if total_queries > 0
                else 0,
                1,
            ),
        },
        "users": users,
        "expensive_queries": expensive_queries,
    }
  except Exception as e:
    return {"error": str(e)}


def generate_recommendations(report: dict) -> list:
  """Generate cost optimization recommendations."""
  recommendations = []

  summary = report.get("summary", {})
  users = report.get("users", [])

  # Check overall cache hit rate
  cache_rate = summary.get("cache_hit_rate", 0)
  if cache_rate < 50:
    recommendations.append({
        "severity": "HIGH",
        "area": "Caching",
        "recommendation": (
            f"Cache hit rate is only {cache_rate}%. Consider using "
            "deterministic queries and avoiding CURRENT_TIMESTAMP() "
            "to improve cache utilization."
        ),
    })

  # Check for high-cost users
  total_cost = summary.get("estimated_cost_usd", 0)
  for user in users[:5]:
    if user["estimated_cost_usd"] > total_cost * 0.3:
      recommendations.append({
          "severity": "MEDIUM",
          "area": "User Cost",
          "recommendation": (
              f"User {user['email']} accounts for"
              f" ${user['estimated_cost_usd']:.2f} ({user['estimated_cost_usd'] / total_cost * 100:.0f}%"
              " of total). Review their query patterns."
          ),
      })

  # Check expensive queries
  for query in report.get("expensive_queries", [])[:3]:
    if query["cost_usd"] > 1:
      recommendations.append({
          "severity": "MEDIUM",
          "area": "Expensive Query",
          "recommendation": (
              f"Query {query['job_id'][:20]}... scanned "
              f"{query['tb_scanned']:.4f} TB (${query['cost_usd']:.2f}). "
              "Consider partitioning or filtering."
          ),
      })

  # General recommendations
  if total_cost > 100:
    recommendations.append({
        "severity": "INFO",
        "area": "Reservations",
        "recommendation": (
            f"With ${total_cost:.2f}/month spend, consider "
            "slot reservations for predictable costs."
        ),
    })

  return recommendations


def format_report(report: dict, recommendations: list) -> str:
  """Format report for display."""
  output = []
  output.append("=" * 70)
  output.append("BIGQUERY COST ANALYSIS REPORT")
  output.append("=" * 70)

  if "error" in report:
    output.append(f"\nError: {report['error']}")
    return "\n".join(output)

  output.append(f"\nProject: {report['project']}")
  output.append(f"Period: Last {report['period_days']} days")
  output.append(f"Generated: {report['generated_at']}")

  summary = report["summary"]
  output.append("\n## Summary")
  output.append(f"  Total Queries: {summary['total_queries']:,}")
  output.append(f"  Total TB Scanned: {summary['total_tb_scanned']:.4f}")
  output.append(f"  Estimated Cost: ${summary['estimated_cost_usd']:.2f}")
  output.append(f"  Unique Users: {summary['unique_users']}")
  output.append(f"  Cache Hit Rate: {summary['cache_hit_rate']:.1f}%")

  output.append("\n## Top Users by Cost")
  output.append("-" * 70)
  output.append(f"{'User':<35} {'Queries':>10} {'TB':>10} {'Cost':>10}")
  output.append("-" * 70)
  for user in report["users"][:10]:
    output.append(
        f"{user['email'][:35]:<35} "
        f"{user['query_count']:>10,} "
        f"{user['tb_scanned']:>10.4f} "
        f"${user['estimated_cost_usd']:>9.2f}"
    )

  output.append("\n## Most Expensive Queries")
  output.append("-" * 70)
  for i, query in enumerate(report["expensive_queries"][:5], 1):
    output.append(f"\n{i}. Job: {query['job_id'][:40]}")
    output.append(f"   User: {query['user']}")
    output.append(
        f"   Cost: ${query['cost_usd']:.2f} ({query['tb_scanned']:.4f} TB)"
    )
    output.append(f"   Query: {query['query_preview'][:60]}...")

  output.append("\n## Recommendations")
  output.append("-" * 70)
  for rec in recommendations:
    output.append(f"\n[{rec['severity']}] {rec['area']}")
    output.append(f"  {rec['recommendation']}")

  output.append("\n" + "=" * 70)
  return "\n".join(output)


def main():
  parser = argparse.ArgumentParser(
      description="Generate BigQuery cost analysis report"
  )
  parser.add_argument("--project", required=True, help="GCP project ID")
  parser.add_argument(
      "--days",
      type=int,
      default=30,
      help="Number of days to analyze (default: 30)",
  )
  parser.add_argument(
      "--format", choices=["text", "json"], default="text", help="Output format"
  )

  args = parser.parse_args()

  report = get_cost_report(args.project, args.days)
  recommendations = generate_recommendations(report)

  if args.format == "json":
    output = {"report": report, "recommendations": recommendations}
    print(json.dumps(output, indent=2, default=str))
  else:
    print(format_report(report, recommendations))


if __name__ == "__main__":
  main()
