"""
BigQuery utilities for data operations
"""

from typing import List, Dict, Any, Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging
from .config import Config

logger = logging.getLogger(__name__)


def get_bq_client() -> bigquery.Client:
    """Get BigQuery client with proper configuration"""
    return bigquery.Client(project=Config.PROJECT_ID)


def ensure_dataset_exists(
    client: bigquery.Client,
    dataset_id: str,
    location: Optional[str] = None
) -> bigquery.Dataset:
    """
    Ensure BigQuery dataset exists, create if not

    Args:
        client: BigQuery client
        dataset_id: Dataset ID (without project prefix)
        location: Dataset location (defaults to Config.BQ_LOCATION)

    Returns:
        Dataset reference
    """
    dataset_ref = client.dataset(dataset_id)

    try:
        dataset = client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_id} already exists")
        return dataset
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location or Config.BQ_LOCATION
        dataset.description = "Security insights data for ADK Security Agent"

        dataset = client.create_dataset(dataset)
        logger.info(f"Created dataset {dataset_id} in {dataset.location}")
        return dataset


def ensure_table_exists(
    client: bigquery.Client,
    table_id: str,
    schema: List[bigquery.SchemaField],
    partition_field: Optional[str] = None
) -> bigquery.Table:
    """
    Ensure BigQuery table exists with schema

    Args:
        client: BigQuery client
        table_id: Fully qualified table ID
        schema: Table schema
        partition_field: Optional field for partitioning

    Returns:
        Table reference
    """
    try:
        table = client.get_table(table_id)
        logger.info(f"Table {table_id} already exists")

        # Check if schema needs update
        existing_fields = {field.name for field in table.schema}
        new_fields = {field.name for field in schema}

        if new_fields - existing_fields:
            # Add new fields
            table.schema = schema
            table = client.update_table(table, ["schema"])
            logger.info(f"Updated schema for table {table_id}")

        return table

    except NotFound:
        table = bigquery.Table(table_id, schema=schema)

        # Add partitioning if specified
        if partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field
            )

        table = client.create_table(table)
        logger.info(f"Created table {table_id}")
        return table


def insert_rows_batch(
    client: bigquery.Client,
    table_id: str,
    rows: List[Dict[str, Any]],
    max_batch_size: int = 500
) -> List[Dict]:
    """
    Insert rows in batches with error handling

    Args:
        client: BigQuery client
        table_id: Fully qualified table ID
        rows: List of row dictionaries
        max_batch_size: Maximum rows per batch

    Returns:
        List of errors (empty if successful)
    """
    if not rows:
        return []

    all_errors = []

    # Process in batches
    for i in range(0, len(rows), max_batch_size):
        batch = rows[i:i + max_batch_size]

        try:
            errors = client.insert_rows_json(table_id, batch)
            if errors:
                all_errors.extend(errors)
                logger.error(f"BigQuery insert errors: {errors}")
        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            all_errors.append({"error": str(e), "batch_start": i})

    if not all_errors:
        logger.info(f"Successfully inserted {len(rows)} rows into {table_id}")

    return all_errors


def create_view(
    client: bigquery.Client,
    view_id: str,
    query: str,
    description: Optional[str] = None
) -> bigquery.Table:
    """
    Create or update a BigQuery view

    Args:
        client: BigQuery client
        view_id: Fully qualified view ID
        query: SQL query for the view
        description: View description

    Returns:
        View reference
    """
    view = bigquery.Table(view_id)
    view.view_query = query

    if description:
        view.description = description

    try:
        # Try to create the view
        view = client.create_table(view)
        logger.info(f"Created view {view_id}")
    except Exception:
        # View exists, update it
        view = client.update_table(view, ["view_query", "description"])
        logger.info(f"Updated view {view_id}")

    return view


def run_query(
    client: bigquery.Client,
    query: str,
    timeout: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Run a BigQuery query and return results

    Args:
        client: BigQuery client
        query: SQL query
        timeout: Query timeout in seconds

    Returns:
        List of result rows as dictionaries
    """
    query_job = client.query(query)
    results = query_job.result(timeout=timeout)

    return [dict(row) for row in results]