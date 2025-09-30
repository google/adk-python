"""
Security-focused BigQuery tools for analyzing GCP security insights
"""

from .base import (
    bq_client, check_client, PROJECT_ID,
    DEFAULT_DATASET, DEFAULT_TABLE, MAX_RESULTS,
    logger
)

def get_security_insights_summary() -> str:
    """
    PRIMARY FUNCTION: Get a comprehensive summary of the security_insights.security_findings table.

    This is the main entry point for analyzing security data. Always call this first
    when users ask about security posture, issues, or general security questions.

    Returns summary statistics from the security_insights dataset including:
    - Total security findings
    - Categories of issues
    - Severity distribution
    - Resource types affected
    - Date range of findings
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    try:
        # First check if the dataset/table exists
        table_ref = bq_client.dataset(dataset_id).table(table_id)
        table = bq_client.get_table(table_ref)

        # Get summary statistics
        query = f"""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT category) as unique_categories,
            COUNT(DISTINCT severity) as severity_levels,
            MIN(created_at) as earliest_record,
            MAX(created_at) as latest_record,
            COUNT(DISTINCT resource_type) as resource_types
        FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
        """

        results = bq_client.query(query).result()

        output = [f"📊 Security Insights Summary ({dataset_id}.{table_id}):"]
        output.append(f"   Table Size: {table.num_rows:,} rows, {table.num_bytes:,} bytes")

        for row in results:
            output.append(f"   Total Records: {row.total_records:,}")
            output.append(f"   Unique Categories: {row.unique_categories}")
            output.append(f"   Severity Levels: {row.severity_levels}")
            output.append(f"   Resource Types: {row.resource_types}")
            output.append(f"   Date Range: {row.earliest_record} to {row.latest_record}")

        return "\n".join(output)
    except Exception as e:
        return f"Error getting security insights summary: {e}"


def query_security_insights(query_filter: str = "", limit: int = 0) -> str:
    """
    Query the security_insights.security_findings table with optional filtering.

    This queries the MAIN SECURITY DATASET that contains all GCP security findings.
    Use this for specific searches like:
    - Filtering by severity (e.g., "severity = 'CRITICAL'")
    - Finding specific resource types
    - Searching for particular categories of issues

    Args:
        query_filter: SQL WHERE clause conditions (e.g., "severity = 'HIGH'")
        limit: Max number of results (default from MAX_RESULTS env var)

    Returns:
        Formatted results from security_insights dataset
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    limit = limit if limit > 0 else MAX_RESULTS
    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    try:
        # Build query with optional filter
        if query_filter:
            query = f"""
            SELECT * FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
            WHERE {query_filter}
            LIMIT {limit}
            """
        else:
            query = f"""
            SELECT * FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
            LIMIT {limit}
            """

        results = bq_client.query(query).result()
        rows = list(results)

        if rows:
            columns = [field.name for field in results.schema]
            output = [f"🔍 Security Insights Query Results:"]
            output.append(f"   Found {len(rows)} record(s)")
            output.append(f"   Columns: {', '.join(columns[:5])}..." if len(columns) > 5 else f"   Columns: {', '.join(columns)}")
            output.append("-" * 50)

            # Show first few rows
            for i, row in enumerate(rows[:10], 1):
                row_dict = dict(row)
                # Format for readability
                output.append(f"\n📌 Record {i}:")
                for key, value in list(row_dict.items())[:8]:  # Show first 8 fields
                    output.append(f"   {key}: {value}")
                if len(row_dict) > 8:
                    output.append(f"   ... and {len(row_dict) - 8} more fields")

            if len(rows) > 10:
                output.append(f"\n... and {len(rows) - 10} more records")

            return "\n".join(output)
        else:
            return "Query executed but no matching records found"
    except Exception as e:
        return f"Error querying security insights: {e}"


def get_security_statistics(group_by: str = "severity") -> str:
    """
    Get aggregated statistics from the security_insights.security_findings table.

    This provides high-level analytics on the security_insights dataset.
    Perfect for understanding the overall security posture and trends.

    Args:
        group_by: Field to group by - severity, category, resource_type, status, or region

    Returns:
        Statistical analysis of security_insights data with counts and percentages
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    dataset_id = DEFAULT_DATASET
    table_id = DEFAULT_TABLE

    # Validate group_by field
    valid_fields = ["severity", "category", "resource_type", "status", "region"]
    if group_by not in valid_fields:
        group_by = "severity"

    try:
        query = f"""
        SELECT
            {group_by},
            COUNT(*) as count,
            COUNT(DISTINCT resource_id) as affected_resources
        FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
        GROUP BY {group_by}
        ORDER BY count DESC
        """

        results = bq_client.query(query).result()
        rows = list(results)

        output = [f"📊 Security Statistics (grouped by {group_by}):"]
        total = sum(row.count for row in rows)
        output.append(f"   Total Records: {total:,}")
        output.append("-" * 50)

        for row in rows:
            percentage = (row.count / total) * 100 if total > 0 else 0
            output.append(f"   {row[group_by] or 'Unknown'}:")
            output.append(f"     - Count: {row.count:,} ({percentage:.1f}%)")
            output.append(f"     - Affected Resources: {row.affected_resources:,}")

        return "\n".join(output)
    except Exception as e:
        return f"Error getting statistics: {e}"


