"""Security-focused BigQuery tools for analyzing GCP security insights."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from typing import Iterable, List

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from .base import (
    DEFAULT_DATASET,
    DEFAULT_TABLE,
    MAX_RESULTS,
    PROJECT_ID,
    StructuredToolResponse,
    bq_client,
    check_client,
    logger,
)


def _error_response(message: str) -> str:
    """Build an error message."""

    logger.error(message)
    return message


def _chunk_rows(rows: Iterable[bigquery.table.Row]) -> List[dict]:
    """Convert BigQuery rows to a list of dictionaries with JSON-safe values."""

    safe_rows: List[dict] = []
    for row in rows:
        row_dict = dict(row)
        safe_rows.append({key: value for key, value in row_dict.items()})
    return safe_rows


def get_security_insights_summary() -> str:
    """Summarize the primary security findings table with structured metrics."""

    try:
        check_client()
    except Exception as exc:  # pragma: no cover - requires missing credentials
        return _error_response(f"Error: {exc}")

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    try:
        table_ref = bq_client.dataset(dataset_id).table(table_id)
        table = bq_client.get_table(table_ref)
    except NotFound:
        return _error_response(
            f"Table {dataset_id}.{table_id} was not found in project {PROJECT_ID}."
        )

    query = f"""
        SELECT
            COUNT(*) AS total_records,
            COUNT(DISTINCT category) AS unique_categories,
            COUNT(DISTINCT severity) AS severity_levels,
            COUNT(DISTINCT resource_name) AS resource_names,
            MIN(created_at) AS earliest_record,
            MAX(created_at) AS latest_record
        FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
    """

    try:
        results = list(bq_client.query(query).result())
    except Exception as exc:  # pragma: no cover - requires live BQ errors
        return _error_response(f"Error getting security insights summary: {exc}")

    metrics = {
        "total_records": 0,
        "unique_categories": 0,
        "severity_levels": 0,
        "resource_names": 0,
        "earliest_record": None,
        "latest_record": None,
    }

    if results:
        row = results[0]
        metrics.update(
            {
                "total_records": row.total_records,
                "unique_categories": row.unique_categories,
                "severity_levels": row.severity_levels,
                "resource_names": row.resource_names,
                "earliest_record": row.earliest_record,
                "latest_record": row.latest_record,
            }
        )

    summary_lines = [
        f"📊 Security Insights Summary ({dataset_id}.{table_id}):",
        f"   Table Size: {table.num_rows:,} rows, {table.num_bytes:,} bytes",
        f"   Total Records: {metrics['total_records']:,}",
        f"   Unique Categories: {metrics['unique_categories']}",
        f"   Severity Levels: {metrics['severity_levels']}",
        f"   Unique Resources: {metrics['resource_names']}",
        "   Date Range: "
        f"{metrics['earliest_record']} to {metrics['latest_record']}",
    ]

    data = {
        "dataset": dataset_id,
        "table": table_id,
        "table_details": {"rows": table.num_rows, "bytes": table.num_bytes},
        "metrics": metrics,
    }

    return "\n".join(summary_lines)


def query_security_insights(query_filter: str = "", limit: int = 0) -> str:
    """Query the security findings table with optional filtering.

    Available columns for filtering:
    - id (INTEGER): Unique identifier
    - name (STRING): Finding name
    - category (STRING): Security category
    - severity (STRING): Severity level (e.g., HIGH, MEDIUM, LOW)
    - resource_name (STRING): Affected resource
    - description (STRING): Finding description
    - recommendation (STRING): Remediation recommendation
    - state (STRING): Current state
    - created_at (STRING): Creation timestamp
    - project_id (STRING): GCP project ID

    Example filters:
    - "severity = 'HIGH'"
    - "category = 'VULNERABILITY'"
    - "created_at >= '2025-10-06'"
    """

    try:
        check_client()
    except Exception as exc:  # pragma: no cover - requires missing credentials
        return _error_response(f"Error: {exc}")

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE
    limit = limit if limit > 0 else MAX_RESULTS

    base_query = f"""
        SELECT * FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
    """
    if query_filter:
        base_query += f"\n        WHERE {query_filter}"
    base_query += f"\n        LIMIT {limit}"

    query_text = base_query.strip()

    try:
        job = bq_client.query(query_text)
        rows_iterable = job.result()
    except Exception as exc:  # pragma: no cover - requires live BQ errors
        return _error_response(f"Error querying security insights: {exc}")

    schema = rows_iterable.schema
    rows = list(rows_iterable)
    columns = [field.name for field in schema] if rows else []

    summary_lines = [
        "🔍 Security Insights Query Results:",
        f"   Found {len(rows)} record(s)",
    ]
    if columns:
        preview_columns = ", ".join(columns[:5])
        summary_lines.append(
            "   Columns: "
            f"{preview_columns}{'...' if len(columns) > 5 else ''}"
        )

    detail_rows = []
    for index, row in enumerate(rows[:10], 1):
        row_dict = dict(row)
        preview_items = list(row_dict.items())[:8]
        detail_lines = [f"📌 Record {index}:"]
        detail_lines.extend(
            f"   {key}: {value}" for key, value in preview_items
        )
        if len(row_dict) > 8:
            detail_lines.append(
                f"   ... and {len(row_dict) - 8} more fields"
            )
        detail_rows.append("\n".join(detail_lines))

    if detail_rows:
        summary_lines.append("-" * 50)
        summary_lines.extend(detail_rows)
    if len(rows) > 10:
        summary_lines.append(f"\n... and {len(rows) - 10} more records")

    return "\n".join(summary_lines)


def get_security_statistics(group_by: str = "severity") -> str:
    """Provide aggregated statistics from the security findings table.

    Valid group_by values:
    - severity: Group by severity level
    - category: Group by security category
    - state: Group by finding state
    - project_id: Group by GCP project
    """

    try:
        check_client()
    except Exception as exc:  # pragma: no cover - requires missing credentials
        return _error_response(f"Error: {exc}")

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    valid_fields = {"severity", "category", "resource_type", "status", "region"}
    if group_by not in valid_fields:
        logger.warning(
            "Invalid group_by '%s' requested. Falling back to 'severity'.", group_by
        )
        group_by = "severity"

    query = f"""
        SELECT
            {group_by} AS grouping_value,
            COUNT(*) AS count,
            COUNT(DISTINCT resource_id) AS affected_resources
        FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
        GROUP BY grouping_value
        ORDER BY count DESC
    """

    try:
        results = list(bq_client.query(query).result())
    except Exception as exc:  # pragma: no cover - requires live BQ errors
        return _error_response(f"Error getting statistics: {exc}")

    total = sum(row.count for row in results)
    distribution = []
    summary_lines = [f"📊 Security Statistics (grouped by {group_by}):"]
    summary_lines.append(f"   Total Records: {total:,}")
    summary_lines.append("-" * 50)

    for row in results:
        label = row.grouping_value or "Unknown"
        percentage = (row.count / total * 100) if total else 0
        summary_lines.append(f"   {label}:")
        summary_lines.append(f"     - Count: {row.count:,} ({percentage:.1f}%)")
        summary_lines.append(
            f"     - Affected Resources: {row.affected_resources:,}"
        )
        distribution.append(
            {
                "value": label,
                "count": row.count,
                "affected_resources": row.affected_resources,
                "percentage": round(percentage, 1),
            }
        )

    data = {
        "group_by": group_by,
        "total_records": total,
        "distribution": distribution,
    }

    return "\n".join(summary_lines)


def get_resources_by_severity(severity: str = "HIGH") -> str:
    """List all unique resources affected by findings of a specific severity level.

    Valid severity values:
    - CRITICAL: Critical security issues requiring immediate attention
    - HIGH: High severity issues that should be addressed soon
    - MEDIUM: Medium severity issues for scheduled remediation
    - LOW: Low severity issues for eventual remediation

    Args:
        severity: The severity level to filter by (default: HIGH)

    Returns:
        Formatted string with list of affected resources and their finding counts
    """
    try:
        check_client()
    except Exception as exc:  # pragma: no cover - requires missing credentials
        return _error_response(f"Error: {exc}")

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    # Validate severity input
    valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    severity_upper = severity.upper()
    if severity_upper not in valid_severities:
        return _error_response(
            f"Invalid severity '{severity}'. Valid options: {', '.join(valid_severities)}"
        )

    query = f"""
        SELECT
            resource_name,
            COUNT(*) AS finding_count,
            STRING_AGG(DISTINCT category, ', ') AS categories,
            MAX(created_at) AS latest_finding
        FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
        WHERE UPPER(severity) = '{severity_upper}'
        GROUP BY resource_name
        ORDER BY finding_count DESC, latest_finding DESC
    """

    try:
        results = list(bq_client.query(query).result())
    except Exception as exc:  # pragma: no cover - requires live BQ errors
        return _error_response(f"Error querying resources by severity: {exc}")

    if not results:
        return f"🔍 No resources found with {severity_upper} severity findings."

    summary_lines = [
        f"🚨 Resources with {severity_upper} Severity Findings:",
        f"   Total Affected Resources: {len(results)}",
        "-" * 50,
    ]

    for idx, row in enumerate(results, 1):
        summary_lines.append(f"\n📌 Resource #{idx}: {row.resource_name}")
        summary_lines.append(f"   Finding Count: {row.finding_count}")
        summary_lines.append(f"   Categories: {row.categories}")
        summary_lines.append(f"   Latest Finding: {row.latest_finding}")

    return "\n".join(summary_lines)


def get_recent_findings(days: int = 7) -> str:
    """Get security findings from the last N days.

    Args:
        days: Number of days to look back (default: 7)

    Returns:
        Formatted string with recent findings grouped by severity
    """
    try:
        check_client()
    except Exception as exc:  # pragma: no cover - requires missing credentials
        return _error_response(f"Error: {exc}")

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    # Validate days input
    if days < 1:
        return _error_response("Days must be a positive number (minimum: 1)")
    if days > 365:
        return _error_response("Days cannot exceed 365")

    # Calculate cutoff date
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    query = f"""
        SELECT
            severity,
            category,
            resource_name,
            name,
            created_at,
            state
        FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
        WHERE created_at >= '{cutoff_date}'
        ORDER BY
            CASE severity
                WHEN 'CRITICAL' THEN 1
                WHEN 'HIGH' THEN 2
                WHEN 'MEDIUM' THEN 3
                WHEN 'LOW' THEN 4
                ELSE 5
            END,
            created_at DESC
    """

    try:
        results = list(bq_client.query(query).result())
    except Exception as exc:  # pragma: no cover - requires live BQ errors
        return _error_response(f"Error querying recent findings: {exc}")

    if not results:
        return f"🔍 No security findings found in the last {days} day(s)."

    summary_lines = [
        f"📅 Security Findings - Last {days} Day(s):",
        f"   Date Range: {cutoff_date} to {datetime.now().strftime('%Y-%m-%d')}",
        f"   Total Findings: {len(results)}",
        "-" * 50,
    ]

    # Group by severity for summary
    severity_counts = {}
    for row in results:
        sev = row.severity or "UNKNOWN"
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    summary_lines.append("\n📊 Severity Breakdown:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        if sev in severity_counts:
            summary_lines.append(f"   {sev}: {severity_counts[sev]}")

    summary_lines.append("\n" + "-" * 50)
    summary_lines.append("\n📋 Recent Findings Details:")

    for idx, row in enumerate(results[:20], 1):  # Show first 20
        summary_lines.append(f"\n{idx}. [{row.severity}] {row.name}")
        summary_lines.append(f"   Resource: {row.resource_name}")
        summary_lines.append(f"   Category: {row.category}")
        summary_lines.append(f"   State: {row.state}")
        summary_lines.append(f"   Date: {row.created_at}")

    if len(results) > 20:
        summary_lines.append(f"\n... and {len(results) - 20} more findings")

    return "\n".join(summary_lines)


def export_findings_to_csv(
    query_filter: str = "", output_file: str = "security_findings.csv"
) -> str:
    """Export security findings to a CSV file.

    Args:
        query_filter: SQL WHERE clause to filter results (optional)
        output_file: Output CSV filename (default: security_findings.csv)

    Returns:
        Success message with file path or error message

    Example:
        export_findings_to_csv("severity = 'HIGH'", "high_severity.csv")
    """
    try:
        check_client()
    except Exception as exc:  # pragma: no cover - requires missing credentials
        return _error_response(f"Error: {exc}")

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    # Ensure .csv extension
    if not output_file.endswith(".csv"):
        output_file += ".csv"

    base_query = f"""
        SELECT * FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
    """
    if query_filter:
        base_query += f"\n        WHERE {query_filter}"
    base_query += "\n        ORDER BY created_at DESC"

    query_text = base_query.strip()

    try:
        job = bq_client.query(query_text)
        rows_iterable = job.result()
    except Exception as exc:  # pragma: no cover - requires live BQ errors
        return _error_response(f"Error querying for export: {exc}")

    rows = list(rows_iterable)
    if not rows:
        return "⚠️ No findings match the criteria. CSV file not created."

    # Get column names from schema
    schema = rows_iterable.schema
    columns = [field.name for field in schema]

    # Write to CSV
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        summary_lines = [
            "✅ Export Successful!",
            f"   File: {output_file}",
            f"   Records Exported: {len(rows)}",
            f"   Columns: {len(columns)}",
            f"   Filter Applied: {query_filter if query_filter else 'None (all records)'}",
        ]
        return "\n".join(summary_lines)

    except Exception as exc:
        return _error_response(f"Error writing CSV file: {exc}")
