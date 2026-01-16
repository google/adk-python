#!/usr/bin/env python3
"""Perform semantic search on BigQuery embeddings.

This script performs semantic search using VECTOR_SEARCH function,
finding similar items based on query text or embeddings.

Usage:
    python semantic_search.py --project-id PROJECT --table TABLE --query "search text"

Examples:
    # Search by text query
    python semantic_search.py \\
        --project-id my-project \\
        --table my_dataset.document_embeddings \\
        --query "machine learning best practices"

    # Search with more results
    python semantic_search.py \\
        --project-id my-project \\
        --table my_dataset.embeddings \\
        --query "data pipeline architecture" \\
        --top-k 20

    # Search with similarity threshold
    python semantic_search.py \\
        --project-id my-project \\
        --table my_dataset.embeddings \\
        --query "API documentation" \\
        --threshold 0.3
"""

import argparse
import json
import sys
from typing import Optional


def generate_search_sql(
    project_id: str,
    embeddings_table: str,
    embedding_model: str,
    query_text: str,
    top_k: int = 10,
    distance_type: str = "COSINE",
    threshold: Optional[float] = None,
    content_column: str = "content",
    id_column: str = "id",
    use_index: bool = True,
    fraction_lists: float = 0.01,
) -> str:
  """Generate SQL for semantic search."""
  # Build options for index usage
  if use_index:
    options = (
        f", options => '{{\"fraction_lists_to_search\": {fraction_lists}}}'"
    )
  else:
    options = ", options => '{\"use_brute_force\": true}'"

  # Build threshold filter
  threshold_filter = ""
  if threshold is not None:
    threshold_filter = f"\nWHERE distance < {threshold}"

  sql = f"""-- Semantic search for: {query_text[:50]}...
WITH query_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{embedding_model}`,
    (SELECT '{query_text.replace("'", "''")}' AS content),
    STRUCT('RETRIEVAL_QUERY' AS task_type)
  )
)
SELECT
  base.{id_column} AS id,
  base.{content_column} AS content,
  distance,
  1.0 - distance AS similarity_score
FROM VECTOR_SEARCH(
  TABLE `{embeddings_table}`,
  'embedding',
  TABLE query_embedding,
  top_k => {top_k},
  distance_type => '{distance_type}'{options}
){threshold_filter}
ORDER BY distance ASC;"""

  return sql


def generate_batch_search_sql(
    project_id: str,
    embeddings_table: str,
    embedding_model: str,
    queries_table: str,
    query_column: str,
    top_k: int = 10,
    distance_type: str = "COSINE",
) -> str:
  """Generate SQL for batch semantic search."""
  sql = f"""-- Batch semantic search
WITH query_embeddings AS (
  SELECT
    query_id,
    query_text,
    ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{embedding_model}`,
    (SELECT query_id, {query_column} AS content FROM `{queries_table}`),
    STRUCT('RETRIEVAL_QUERY' AS task_type)
  )
  WHERE LENGTH(ml_generate_embedding_status) = 0
)
SELECT
  query.query_id,
  query.query_text,
  base.id,
  base.content,
  distance,
  1.0 - distance AS similarity_score
FROM VECTOR_SEARCH(
  TABLE `{embeddings_table}`,
  'embedding',
  TABLE query_embeddings,
  top_k => {top_k},
  distance_type => '{distance_type}'
)
ORDER BY query.query_id, distance ASC;"""

  return sql


def main():
  parser = argparse.ArgumentParser(
      description="Perform semantic search on BigQuery embeddings",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )

  parser.add_argument(
      "--project-id",
      required=True,
      help="Google Cloud project ID",
  )
  parser.add_argument(
      "--table",
      required=True,
      help="Embeddings table (dataset.table)",
  )
  parser.add_argument(
      "--query",
      help="Search query text",
  )
  parser.add_argument(
      "--queries-table",
      help="Table containing multiple queries for batch search",
  )
  parser.add_argument(
      "--query-column",
      default="query_text",
      help="Column name for query text in batch mode (default: query_text)",
  )
  parser.add_argument(
      "--embedding-model",
      help="Embedding model to use (default: auto-detect or create)",
  )
  parser.add_argument(
      "--top-k",
      type=int,
      default=10,
      help="Number of results to return (default: 10)",
  )
  parser.add_argument(
      "--threshold",
      type=float,
      help="Maximum distance threshold (0.0-1.0 for COSINE)",
  )
  parser.add_argument(
      "--distance-type",
      choices=["COSINE", "EUCLIDEAN", "DOT_PRODUCT"],
      default="COSINE",
      help="Distance metric (default: COSINE)",
  )
  parser.add_argument(
      "--content-column",
      default="content",
      help="Column containing text content (default: content)",
  )
  parser.add_argument(
      "--id-column",
      default="id",
      help="Column containing ID (default: id)",
  )
  parser.add_argument(
      "--brute-force",
      action="store_true",
      help="Use brute force search (exact but slower)",
  )
  parser.add_argument(
      "--fraction-lists",
      type=float,
      default=0.01,
      help="Fraction of index lists to search (default: 0.01)",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print SQL without executing",
  )
  parser.add_argument(
      "--output",
      choices=["table", "json", "sql"],
      default="table",
      help="Output format (default: table)",
  )

  args = parser.parse_args()

  # Validate inputs
  if not args.query and not args.queries_table:
    print("Error: Either --query or --queries-table must be specified")
    return 1

  # Set defaults
  dataset = args.table.split(".")[0]
  embedding_model = (
      args.embedding_model
      or f"{args.project_id}.{dataset}.text_embedding_model"
  )

  # Generate SQL
  if args.queries_table:
    sql = generate_batch_search_sql(
        project_id=args.project_id,
        embeddings_table=f"{args.project_id}.{args.table}",
        embedding_model=embedding_model,
        queries_table=f"{args.project_id}.{args.queries_table}",
        query_column=args.query_column,
        top_k=args.top_k,
        distance_type=args.distance_type,
    )
  else:
    sql = generate_search_sql(
        project_id=args.project_id,
        embeddings_table=f"{args.project_id}.{args.table}",
        embedding_model=embedding_model,
        query_text=args.query,
        top_k=args.top_k,
        distance_type=args.distance_type,
        threshold=args.threshold,
        content_column=args.content_column,
        id_column=args.id_column,
        use_index=not args.brute_force,
        fraction_lists=args.fraction_lists,
    )

  if args.output == "sql" or args.dry_run:
    print(sql)
    if args.dry_run:
      print("\n-- Dry run mode: SQL not executed")
    return 0

  # Execute the SQL
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=args.project_id)

    print(
        "Searching for:"
        f" {args.query[:50] if args.query else 'batch queries'}..."
    )
    query_job = client.query(sql)
    results = list(query_job.result())

    if not results:
      print("No results found.")
      return 0

    if args.output == "json":
      output = []
      for row in results:
        output.append({
            "id": row.id,
            "content": (
                row.content[:200] + "..."
                if len(row.content) > 200
                else row.content
            ),
            "distance": row.distance,
            "similarity_score": row.similarity_score,
        })
      print(json.dumps(output, indent=2))
    else:
      # Table output
      print(f"\n{'Rank':<6}{'ID':<20}{'Similarity':<12}{'Content':<60}")
      print("-" * 98)
      for i, row in enumerate(results, 1):
        content_preview = (
            row.content[:57] + "..." if len(row.content) > 60 else row.content
        )
        content_preview = content_preview.replace("\n", " ")
        print(
            f"{i:<6}{str(row.id)[:18]:<20}{row.similarity_score:.4f}     "
            f" {content_preview}"
        )

      print(f"\nFound {len(results)} results")

    return 0

  except ImportError:
    print("\n-- Note: google-cloud-bigquery not installed")
    print("-- To execute, install it with: pip install google-cloud-bigquery")
    print("-- Here's the SQL to run manually:")
    print()
    print(sql)
    return 0

  except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
