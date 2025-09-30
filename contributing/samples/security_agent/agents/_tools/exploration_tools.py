"""
Data exploration and analysis tools for BigQuery
"""

from .base import (
    bq_client, check_client, PROJECT_ID,
    DEFAULT_DATASET, DEFAULT_TABLE,
    ENABLE_EXPLORATION, logger
)

def explore_all_tables_and_views(dataset_id: str = "") -> str:
    """List all tables AND views in a dataset, distinguishing between them"""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    if not ENABLE_EXPLORATION and dataset_id != DEFAULT_DATASET:
        return f"Dataset exploration is disabled. Only {DEFAULT_DATASET} is accessible."

    dataset_id = dataset_id if dataset_id else DEFAULT_DATASET

    try:
        tables = list(bq_client.list_tables(dataset_id))

        if not tables:
            return f"No tables or views found in dataset '{dataset_id}'"

        # Categorize tables and views
        regular_tables = []
        views = []
        external_tables = []

        for table in tables:
            table_ref = bq_client.dataset(dataset_id).table(table.table_id)
            table_obj = bq_client.get_table(table_ref)

            table_info = {
                'id': table.table_id,
                'type': table_obj.table_type,
                'rows': table_obj.num_rows,
                'bytes': table_obj.num_bytes,
                'created': table_obj.created,
                'modified': table_obj.modified,
                'description': table_obj.description
            }

            if table_obj.table_type == "TABLE":
                regular_tables.append(table_info)
            elif table_obj.table_type == "VIEW":
                views.append(table_info)
            elif table_obj.table_type == "EXTERNAL":
                external_tables.append(table_info)

        output = [f"📂 Dataset '{dataset_id}' Contents:"]
        output.append(f"   Total Objects: {len(tables)}")

        if regular_tables:
            output.append(f"\n📊 Tables ({len(regular_tables)}):")
            for t in regular_tables:
                output.append(f"   • {t['id']}")
                if t['rows']:
                    output.append(f"     - {t['rows']:,} rows, {t['bytes']:,} bytes")
                if t['description']:
                    output.append(f"     - {t['description']}")

        if views:
            output.append(f"\n👁️ Views ({len(views)}):")
            for v in views:
                output.append(f"   • {v['id']}")
                if v['description']:
                    output.append(f"     - {v['description']}")

        if external_tables:
            output.append(f"\n🔗 External Tables ({len(external_tables)}):")
            for e in external_tables:
                output.append(f"   • {e['id']}")
                if e['description']:
                    output.append(f"     - {e['description']}")

        return "\n".join(output)
    except Exception as e:
        return f"Error exploring dataset: {e}"


def analyze_table_or_view(dataset_id: str, object_id: str) -> str:
    """Analyze a table or view to understand its structure and content"""
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    try:
        table_ref = bq_client.dataset(dataset_id).table(object_id)
        table = bq_client.get_table(table_ref)

        output = [f"📋 Analysis of {dataset_id}.{object_id}:"]
        output.append(f"   Type: {table.table_type}")

        if table.table_type == "VIEW":
            output.append(f"   View Definition: [Stored SQL query]")
            # For views, we can get column info but not row count
            output.append(f"   Columns: {len(table.schema)}")
        else:
            output.append(f"   Rows: {table.num_rows:,}" if table.num_rows else "   Rows: 0")
            output.append(f"   Size: {table.num_bytes:,} bytes" if table.num_bytes else "   Size: 0 bytes")

        output.append(f"   Created: {table.created}")
        output.append(f"   Last Modified: {table.modified}")

        if table.description:
            output.append(f"   Description: {table.description}")

        # Schema details
        output.append("\n📐 Schema:")
        for field in table.schema[:15]:  # Show first 15 fields
            field_type = field.field_type
            mode = field.mode or "NULLABLE"
            desc = f" - {field.description}" if field.description else ""
            output.append(f"   • {field.name}: {field_type} ({mode}){desc}")

        if len(table.schema) > 15:
            output.append(f"   ... and {len(table.schema) - 15} more fields")

        # For tables, get a sample
        if table.table_type == "TABLE":
            output.append(f"\n🔍 Sample Data Preview:")
            sample_query = f"""
            SELECT * FROM `{PROJECT_ID}.{dataset_id}.{object_id}`
            LIMIT 3
            """
            try:
                results = bq_client.query(sample_query).result()
                rows = list(results)
                if rows:
                    for i, row in enumerate(rows, 1):
                        output.append(f"   Row {i}: {dict(row)}")
                else:
                    output.append("   No data available")
            except Exception as e:
                output.append(f"   Could not retrieve sample: {e}")

        return "\n".join(output)
    except Exception as e:
        return f"Error analyzing object: {e}"


