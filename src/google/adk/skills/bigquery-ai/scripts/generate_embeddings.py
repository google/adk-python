#!/usr/bin/env python3
"""Generate embeddings for a BigQuery table.

This script generates vector embeddings for text data in BigQuery using
ML.GENERATE_EMBEDDING, with support for batching and incremental updates.

Usage:
    python generate_embeddings.py --project-id PROJECT --source-table TABLE

Examples:
    # Generate embeddings for a table
    python generate_embeddings.py \\
        --project-id my-project \\
        --source-table my_dataset.documents \\
        --content-column body \\
        --output-table my_dataset.document_embeddings

    # Incremental update (only embed new rows)
    python generate_embeddings.py \\
        --project-id my-project \\
        --source-table my_dataset.documents \\
        --output-table my_dataset.document_embeddings \\
        --incremental --id-column doc_id

    # Custom embedding model
    python generate_embeddings.py \\
        --project-id my-project \\
        --source-table my_dataset.documents \\
        --embedding-model my_dataset.custom_embedding_model
"""

import argparse
import json
import sys
from typing import Optional


def generate_embedding_sql(
    project_id: str,
    source_table: str,
    output_table: str,
    content_column: str,
    id_column: str,
    embedding_model: str,
    task_type: Optional[str] = None,
    batch_size: Optional[int] = None,
    incremental: bool = False,
    create_table: bool = True,
) -> str:
    """Generate SQL for embedding generation."""
    # Build task type option
    task_option = ""
    if task_type:
        task_option = f", STRUCT('{task_type}' AS task_type)"

    # Build source query
    if incremental:
        source_query = f"""
    SELECT s.{id_column}, s.{content_column} AS content
    FROM `{source_table}` s
    LEFT JOIN `{output_table}` e ON s.{id_column} = e.{id_column}
    WHERE e.{id_column} IS NULL"""
    else:
        source_query = f"""
    SELECT {id_column}, {content_column} AS content
    FROM `{source_table}`"""

    if batch_size:
        source_query += f"\n    LIMIT {batch_size}"

    # Build the main SQL
    if create_table and not incremental:
        create_clause = f"CREATE OR REPLACE TABLE `{output_table}` AS"
    else:
        create_clause = f"INSERT INTO `{output_table}`"

    sql = f"""{create_clause}
SELECT
  {id_column},
  content,
  ml_generate_embedding_result AS embedding,
  ml_generate_embedding_status AS status,
  CURRENT_TIMESTAMP() AS embedded_at
FROM ML.GENERATE_EMBEDDING(
  MODEL `{embedding_model}`,
  ({source_query}){task_option}
)
WHERE LENGTH(ml_generate_embedding_status) = 0;"""

    return sql


def generate_index_sql(
    output_table: str,
    index_name: str,
    distance_type: str = "COSINE",
    num_lists: int = 500,
) -> str:
    """Generate SQL for vector index creation."""
    return f"""CREATE OR REPLACE VECTOR INDEX {index_name}
ON `{output_table}`(embedding)
OPTIONS (
  index_type = 'IVF',
  distance_type = '{distance_type}',
  ivf_options = '{{"num_lists": {num_lists}}}'
);"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for BigQuery table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--project-id",
        required=True,
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--source-table",
        required=True,
        help="Source table (dataset.table)",
    )
    parser.add_argument(
        "--output-table",
        help="Output table for embeddings (default: source_table_embeddings)",
    )
    parser.add_argument(
        "--content-column",
        default="content",
        help="Column containing text to embed (default: content)",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Primary key column (default: id)",
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model to use (default: creates text-embedding-005)",
    )
    parser.add_argument(
        "--task-type",
        choices=[
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
        ],
        default="RETRIEVAL_DOCUMENT",
        help="Task type for embeddings (default: RETRIEVAL_DOCUMENT)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Process in batches of this size",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only embed rows not already in output table",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create vector index after embedding",
    )
    parser.add_argument(
        "--index-name",
        help="Name for vector index (default: table_embedding_idx)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL without executing",
    )
    parser.add_argument(
        "--output",
        choices=["sql", "json"],
        default="sql",
        help="Output format (default: sql)",
    )

    args = parser.parse_args()

    # Set defaults
    output_table = args.output_table or f"{args.source_table}_embeddings"
    embedding_model = args.embedding_model or f"{args.project_id}.{args.source_table.split('.')[0]}.text_embedding_model"
    index_name = args.index_name or f"{output_table.split('.')[-1]}_embedding_idx"

    # Generate SQL statements
    sql_statements = []

    # Create embedding model if not specified
    if not args.embedding_model:
        dataset = args.source_table.split(".")[0]
        model_sql = f"""-- Create embedding model if not exists
CREATE MODEL IF NOT EXISTS `{args.project_id}.{dataset}.text_embedding_model`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-embedding-005');"""
        sql_statements.append(model_sql)

    # Generate embeddings
    embed_sql = generate_embedding_sql(
        project_id=args.project_id,
        source_table=f"{args.project_id}.{args.source_table}",
        output_table=f"{args.project_id}.{output_table}",
        content_column=args.content_column,
        id_column=args.id_column,
        embedding_model=embedding_model,
        task_type=args.task_type,
        batch_size=args.batch_size,
        incremental=args.incremental,
        create_table=not args.incremental,
    )
    sql_statements.append(embed_sql)

    # Create index if requested
    if args.create_index:
        index_sql = generate_index_sql(
            output_table=f"{args.project_id}.{output_table}",
            index_name=index_name,
        )
        sql_statements.append(index_sql)

    # Output results
    full_sql = "\n\n".join(sql_statements)

    if args.output == "json":
        result = {
            "project_id": args.project_id,
            "source_table": args.source_table,
            "output_table": output_table,
            "embedding_model": embedding_model,
            "task_type": args.task_type,
            "incremental": args.incremental,
            "create_index": args.create_index,
            "sql": full_sql,
            "dry_run": args.dry_run,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"-- Generate embeddings for {args.source_table}")
        print(f"-- Output: {output_table}")
        print(f"-- Model: {embedding_model}")
        print(f"-- Task type: {args.task_type}")
        if args.incremental:
            print("-- Mode: Incremental (new rows only)")
        print()
        print(full_sql)
        print()

    if args.dry_run:
        print("-- Dry run mode: SQL not executed")
        return 0

    # Execute the SQL
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=args.project_id)

        for i, sql in enumerate(sql_statements):
            print(f"\nExecuting statement {i + 1}/{len(sql_statements)}...")
            query_job = client.query(sql)
            result = query_job.result()
            if hasattr(result, "total_rows"):
                print(f"  Processed {result.total_rows} rows")

        print("\nEmbedding generation complete!")

        # Get stats
        stats_query = f"""
        SELECT
          COUNT(*) as total_embeddings,
          MAX(embedded_at) as last_updated
        FROM `{args.project_id}.{output_table}`
        """
        stats = list(client.query(stats_query).result())[0]
        print(f"Total embeddings: {stats.total_embeddings}")
        print(f"Last updated: {stats.last_updated}")

        return 0

    except ImportError:
        print("\n-- Note: google-cloud-bigquery not installed")
        print("-- To execute, install it with: pip install google-cloud-bigquery")
        print("-- Or run the SQL manually in BigQuery Console")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
