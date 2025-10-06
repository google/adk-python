"""Security-focused BigQuery tools for analyzing GCP security insights."""

from __future__ import annotations

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


def _error_response(message: str) -> StructuredToolResponse:
    """Build a structured error payload."""

    logger.error(message)
    return StructuredToolResponse(summary=message, data={}, metadata={"error": True})


def _chunk_rows(rows: Iterable[bigquery.table.Row]) -> List[dict]:
    """Convert BigQuery rows to a list of dictionaries with JSON-safe values."""

    safe_rows: List[dict] = []
    for row in rows:
        row_dict = dict(row)
        safe_rows.append({key: value for key, value in row_dict.items()})
    return safe_rows


def get_security_insights_summary() -> StructuredToolResponse:
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
            COUNT(DISTINCT resource_type) AS resource_types,
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
        "resource_types": 0,
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
                "resource_types": row.resource_types,
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
        f"   Resource Types: {metrics['resource_types']}",
        "   Date Range: "
        f"{metrics['earliest_record']} to {metrics['latest_record']}",
    ]

    data = {
        "dataset": dataset_id,
        "table": table_id,
        "table_details": {"rows": table.num_rows, "bytes": table.num_bytes},
        "metrics": metrics,
    }

    return StructuredToolResponse(
        summary="\n".join(summary_lines),
        data=data,
        metadata={"query": query.strip()},
    )


def query_security_insights(query_filter: str = "", limit: int = 0) -> StructuredToolResponse:
    """Query the security findings table with optional filtering."""

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

    data = {
        "dataset": dataset_id,
        "table": table_id,
        "row_count": len(rows),
        "columns": columns,
        "records": _chunk_rows(rows[:limit]),
    }

    return StructuredToolResponse(
        summary="\n".join(summary_lines),
        data=data,
        metadata={"query": query_text},
    )


def get_security_statistics(group_by: str = "severity") -> StructuredToolResponse:
    """Provide aggregated statistics from the security findings table."""

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

    return StructuredToolResponse(
        summary="\n".join(summary_lines),
        data=data,
        metadata={"query": query.strip()},
    )
