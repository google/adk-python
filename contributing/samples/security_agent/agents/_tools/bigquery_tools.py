"""Standard BigQuery operations and utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from google.cloud import bigquery

from .base import (
    DEFAULT_DATASET,
    DEFAULT_TABLE,
    MAX_RESULTS,
    PROJECT_ID,
    bq_client,
    check_client,
    logger,
)
from .service_evaluation.evaluator import evaluate_new_service

SAMPLE_LIMIT = 20

def hello_world() -> str:
    """Execute a Hello World query in BigQuery"""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    query = """
    SELECT
        'Hello from Security-Focused BigQuery Agent!' as greeting,
        CURRENT_TIMESTAMP() as timestamp,
        @@project_id as project_id,
        ARRAY<STRING>['Security', 'BigQuery', 'ADK'] as tags
    """

    try:
        results = bq_client.query(query).result()
        output = ["🎉 Security Agent Ready:"]
        for row in results:
            output.append(f"  Greeting: {row.greeting}")
            output.append(f"  Time: {row.timestamp}")
            output.append(f"  Project: {row.project_id}")
            output.append(f"  Tags: {', '.join(row.tags)}")
            output.append(f"  Default Dataset: {DEFAULT_DATASET}")
        return "\n".join(output)
    except Exception as e:
        return f"Error: {e}"


def list_datasets() -> str:
    """List all BigQuery datasets in the project"""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    try:
        datasets = list(bq_client.list_datasets())
        if datasets:
            output = [f"Found {len(datasets)} datasets in project {PROJECT_ID}:"]
            for dataset in datasets:
                ds_id = dataset.dataset_id
                # Highlight the default dataset
                marker = "⭐" if ds_id == DEFAULT_DATASET else "📁"
                # Get dataset info
                ds_ref = bq_client.dataset(ds_id)
                ds_obj = bq_client.get_dataset(ds_ref)
                output.append(f"\n{marker} {ds_id}")
                output.append(f"   Description: {ds_obj.description or 'No description'}")
                output.append(f"   Location: {ds_obj.location}")
                output.append(f"   Created: {ds_obj.created}")
        else:
            output = [f"No datasets found in project {PROJECT_ID}"]
        return "\n".join(output)
    except Exception as e:
        return f"Error listing datasets: {e}"


def list_tables(dataset_id: str = "") -> str:
    """List all tables in a specific BigQuery dataset (defaults to configured dataset)"""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    dataset_id = dataset_id if dataset_id else DEFAULT_DATASET

    try:
        tables = list(bq_client.list_tables(dataset_id))
        if tables:
            output = [f"Found {len(tables)} tables in dataset '{dataset_id}':"]
            for table in tables:
                table_ref = bq_client.dataset(dataset_id).table(table.table_id)
                table_obj = bq_client.get_table(table_ref)
                marker = "⭐" if table.table_id == DEFAULT_TABLE else "📊"
                output.append(f"\n{marker} {table.table_id}")
                output.append(f"   Type: {table_obj.table_type}")
                output.append(f"   Rows: {table_obj.num_rows:,}" if table_obj.num_rows else "   Rows: 0")
                output.append(f"   Size: {table_obj.num_bytes:,} bytes" if table_obj.num_bytes else "   Size: 0 bytes")
        else:
            output = [f"No tables found in dataset '{dataset_id}'"]
        return "\n".join(output)
    except Exception as e:
        return f"Error listing tables: {e}"


def get_table_schema(dataset_id: str = "", table_id: str = "") -> str:
    """Get the schema of a specific BigQuery table"""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    dataset_id = dataset_id if dataset_id else DEFAULT_DATASET
    table_id = table_id if table_id else DEFAULT_TABLE

    try:
        table_ref = bq_client.dataset(dataset_id).table(table_id)
        table = bq_client.get_table(table_ref)

        output = [f"Schema for {dataset_id}.{table_id}:"]
        output.append(f"Total fields: {len(table.schema)}")
        output.append("\nFields:")

        for field in table.schema:
            field_type = field.field_type
            mode = field.mode or "NULLABLE"
            desc = f" - {field.description}" if field.description else ""
            output.append(f"  • {field.name}: {field_type} ({mode}){desc}")

        return "\n".join(output)
    except Exception as e:
        return f"Error getting schema: {e}"


def run_query(query: str) -> str:
    """Execute a general-purpose BigQuery SQL query.

    Note: For security-related queries, prefer the more specific tools like
    `query_security_insights` or `get_security_statistics` when possible.
    This tool is best for queries that are not covered by other tools.
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    try:
        # Run query
        query_job = bq_client.query(query)
        results = query_job.result()

        # Convert to list for counting and display
        rows = list(results)

        if rows:
            # Get column names from the schema
            columns = [field.name for field in results.schema]

            output = [f"Query returned {len(rows)} row(s)"]
            output.append(f"Columns: {', '.join(columns)}")
            output.append("-" * 50)

            # Show first rows based on MAX_RESULTS
            display_limit = min(len(rows), 10)
            for i, row in enumerate(rows[:display_limit], 1):
                row_dict = dict(row)
                output.append(f"Row {i}: {row_dict}")

            if len(rows) > display_limit:
                output.append(f"\n... and {len(rows) - display_limit} more rows")

            # Add query statistics
            output.append(f"\n📊 Query Statistics:")
            output.append(f"  Bytes processed: {query_job.total_bytes_processed:,}")
            output.append(f"  Execution time: {query_job.ended - query_job.started}")

            return "\n".join(output)
        else:
            return "Query executed successfully but returned no rows"

    except GoogleCloudError as e:
        return f"BigQuery error: {e.message if hasattr(e, 'message') else str(e)}"
    except Exception as e:
        return f"Error executing query: {e}"


def analyze_query_cost(query: str) -> str:
    """Analyze the estimated cost and data processed for a query without running it"""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    try:
        # Create a query job config with dry_run
        job_config = bigquery.QueryJobConfig(dry_run=True)

        # Run dry run
        query_job = bq_client.query(query, job_config=job_config)

        # Calculate estimated cost (BigQuery charges $5 per TB as of 2024)
        bytes_processed = query_job.total_bytes_processed
        tb_processed = bytes_processed / (1024 ** 4)
        estimated_cost = tb_processed * 5.00

        output = ["📊 Query Analysis (Dry Run):"]
        output.append(f"  Bytes to process: {bytes_processed:,}")
        output.append(f"  GB to process: {bytes_processed / (1024**3):.3f}")
        output.append(f"  Estimated cost: ${estimated_cost:.4f}")
        output.append("  Status: ✅ Query is valid and ready to run")

        return "\n".join(output)

    except Exception as e:
        return f"Error analyzing query: {e}"


def get_table_sample(dataset_id: str = "", table_id: str = "", limit: int = 0) -> str:
    """Get a sample of rows from a generic BigQuery table."""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    dataset_id = dataset_id if dataset_id else DEFAULT_DATASET
    table_id = table_id if table_id else DEFAULT_TABLE
    limit = limit if limit > 0 else SAMPLE_LIMIT

    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
    LIMIT {limit}
    """

    return run_query(query)


class BigQueryTool:
    """Convenience wrapper for structured BigQuery interactions."""

    def __init__(
        self,
        project_id: str = PROJECT_ID,
        default_dataset: str = DEFAULT_DATASET,
        default_table: str = DEFAULT_TABLE,
    ) -> None:
        self.project_id = project_id
        self.default_dataset = default_dataset
        self.default_table = default_table

    def execute_query(
        self, query: str, *, max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a SQL query and return JSON-serialisable results."""

        try:
            check_client()
        except Exception as exc:  # pragma: no cover - depends on credentials
            return {"success": False, "error": str(exc), "query": query}

        try:
            job = bq_client.query(query)
            results = list(job.result())
        except Exception as exc:  # pragma: no cover - surfaces runtime errors
            logger.exception("BigQueryTool execute_query failed")
            return {"success": False, "error": str(exc), "query": query}

        rows = [dict(row) for row in results]
        if max_results is not None:
            rows = rows[:max_results]

        metadata: Dict[str, Any] = {
            "project_id": self.project_id,
            "default_dataset": self.default_dataset,
            "row_count": len(rows),
            "bytes_processed": getattr(job, "total_bytes_processed", None),
            "slot_millis": getattr(job, "slot_millis", None),
        }

        return {
            "success": True,
            "query": query,
            "data": rows,
            "metadata": metadata,
        }

    def evaluate_service(
        self,
        service_name: str,
        service_type: str,
        service_profile: Optional[str] = None,
        use_case: Optional[str] = None,
        data_classification: Optional[str] = None,
        check_current_compliance: bool = False,
    ) -> Dict[str, Any]:
        """Delegate to the service evaluation framework with structured output."""

        result = evaluate_new_service(
            service_name=service_name,
            service_type=service_type,
            service_profile=service_profile,
            use_case=use_case,
            data_classification=data_classification,
            check_current_compliance=check_current_compliance,
            return_format="dict",
        )

        return {
            "success": True,
            "data": result,
        }
