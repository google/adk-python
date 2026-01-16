#!/usr/bin/env python3
"""Build and run RAG (Retrieval-Augmented Generation) pipelines in BigQuery.

This script helps set up and execute RAG workflows that combine semantic
search with text generation for grounded AI responses.

Usage:
    python rag_pipeline.py --project-id PROJECT --kb-table TABLE --query "question"

Examples:
    # Ask a question using RAG
    python rag_pipeline.py \\
        --project-id my-project \\
        --kb-table my_dataset.knowledge_base_embeddings \\
        --query "What is the refund policy?"

    # Setup new RAG pipeline
    python rag_pipeline.py \\
        --project-id my-project \\
        --source-table my_dataset.documents \\
        --setup

    # RAG with custom models
    python rag_pipeline.py \\
        --project-id my-project \\
        --kb-table my_dataset.kb_embeddings \\
        --query "How do I configure logging?" \\
        --generation-model my_dataset.gemini_pro \\
        --num-sources 10
"""

import argparse
import json
import sys
from typing import Optional


def generate_setup_sql(
    project_id: str,
    source_table: str,
    kb_table: str,
    content_column: str,
    id_column: str,
    chunk_size: int = 1000,
) -> str:
  """Generate SQL to set up a RAG knowledge base."""
  dataset = source_table.split(".")[0]

  sql = f"""-- RAG Pipeline Setup
-- Step 1: Create embedding model
CREATE MODEL IF NOT EXISTS `{project_id}.{dataset}.rag_embedding_model`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-embedding-005');

-- Step 2: Create generation model
CREATE MODEL IF NOT EXISTS `{project_id}.{dataset}.rag_generation_model`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'gemini-2.0-flash');

-- Step 3: Chunk documents (if needed)
CREATE OR REPLACE TABLE `{project_id}.{dataset}.document_chunks` AS
WITH chunks AS (
  SELECT
    {id_column} AS doc_id,
    chunk_index,
    TRIM(chunk) AS chunk_text
  FROM `{project_id}.{source_table}`,
    UNNEST(REGEXP_EXTRACT_ALL({content_column}, r'.{{{{{chunk_size}}}}}(?:\\s|$)|.+$')) AS chunk
    WITH OFFSET AS chunk_index
)
SELECT
  CONCAT(doc_id, '_', chunk_index) AS chunk_id,
  doc_id,
  chunk_index,
  chunk_text
FROM chunks
WHERE LENGTH(chunk_text) > 50;

-- Step 4: Generate embeddings
CREATE OR REPLACE TABLE `{project_id}.{kb_table}` AS
SELECT
  chunk_id,
  doc_id,
  chunk_text AS content,
  ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `{project_id}.{dataset}.rag_embedding_model`,
  (SELECT chunk_id, doc_id, chunk_text AS content
   FROM `{project_id}.{dataset}.document_chunks`),
  STRUCT('RETRIEVAL_DOCUMENT' AS task_type)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;

-- Step 5: Create vector index
CREATE OR REPLACE VECTOR INDEX kb_embedding_idx
ON `{project_id}.{kb_table}`(embedding)
OPTIONS (
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{{"num_lists": 500}}'
);

-- Setup complete! Use rag_pipeline.py --query to ask questions.
SELECT
  'RAG pipeline setup complete' AS status,
  (SELECT COUNT(*) FROM `{project_id}.{kb_table}`) AS total_chunks;"""

  return sql


def generate_rag_query_sql(
    project_id: str,
    kb_table: str,
    embedding_model: str,
    generation_model: str,
    query_text: str,
    num_sources: int = 5,
    max_output_tokens: int = 512,
    temperature: float = 0.2,
    include_sources: bool = True,
) -> str:
  """Generate SQL for RAG query."""
  escaped_query = query_text.replace("'", "''")

  sources_select = ""
  if include_sources:
    sources_select = """,
  (SELECT ARRAY_AGG(STRUCT(
    chunk_id,
    LEFT(content, 200) AS excerpt,
    ROUND(1.0 - distance, 3) AS relevance
   ))
   FROM retrieved_context) AS sources"""

  sql = f"""-- RAG Query: {query_text[:50]}...
DECLARE user_query STRING DEFAULT '{escaped_query}';

WITH query_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{embedding_model}`,
    (SELECT user_query AS content),
    STRUCT('RETRIEVAL_QUERY' AS task_type)
  )
),
retrieved_context AS (
  SELECT
    base.chunk_id,
    base.content,
    distance
  FROM VECTOR_SEARCH(
    TABLE `{kb_table}`,
    'embedding',
    TABLE query_embedding,
    top_k => {num_sources},
    distance_type => 'COSINE',
    options => '{{"fraction_lists_to_search": 0.01}}'
  )
  ORDER BY distance
),
context_string AS (
  SELECT STRING_AGG(
    CONCAT('[Source ', ROW_NUMBER() OVER (ORDER BY distance), ']: ', content),
    '\\n\\n'
  ) AS context
  FROM retrieved_context
),
rag_prompt AS (
  SELECT CONCAT(
    'You are a helpful assistant. Answer the question based ONLY on the following context. ',
    'If the answer is not in the context, say "I don\\'t have enough information to answer that."\\n\\n',
    'Context:\\n', context, '\\n\\n',
    'Question: ', user_query, '\\n\\n',
    'Answer:'
  ) AS prompt
  FROM context_string
)
SELECT
  user_query AS question,
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS answer{sources_select}
FROM ML.GENERATE_TEXT(
  MODEL `{generation_model}`,
  TABLE rag_prompt,
  STRUCT({max_output_tokens} AS max_output_tokens, {temperature} AS temperature)
),
retrieved_context;"""

  return sql


def main():
  parser = argparse.ArgumentParser(
      description="Build and run RAG pipelines in BigQuery",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )

  parser.add_argument(
      "--project-id",
      required=True,
      help="Google Cloud project ID",
  )
  parser.add_argument(
      "--kb-table",
      help="Knowledge base embeddings table (dataset.table)",
  )
  parser.add_argument(
      "--query",
      help="Question to answer using RAG",
  )
  parser.add_argument(
      "--setup",
      action="store_true",
      help="Setup new RAG pipeline from source documents",
  )
  parser.add_argument(
      "--source-table",
      help="Source documents table for setup (dataset.table)",
  )
  parser.add_argument(
      "--content-column",
      default="content",
      help="Column containing document text (default: content)",
  )
  parser.add_argument(
      "--id-column",
      default="id",
      help="Column containing document ID (default: id)",
  )
  parser.add_argument(
      "--chunk-size",
      type=int,
      default=1000,
      help="Characters per chunk (default: 1000)",
  )
  parser.add_argument(
      "--embedding-model",
      help="Embedding model (default: auto-detect)",
  )
  parser.add_argument(
      "--generation-model",
      help="Generation model (default: auto-detect)",
  )
  parser.add_argument(
      "--num-sources",
      type=int,
      default=5,
      help="Number of sources to retrieve (default: 5)",
  )
  parser.add_argument(
      "--max-tokens",
      type=int,
      default=512,
      help="Max output tokens (default: 512)",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=0.2,
      help="Generation temperature (default: 0.2)",
  )
  parser.add_argument(
      "--no-sources",
      action="store_true",
      help="Don't include source references in output",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print SQL without executing",
  )
  parser.add_argument(
      "--output",
      choices=["text", "json", "sql"],
      default="text",
      help="Output format (default: text)",
  )

  args = parser.parse_args()

  # Handle setup mode
  if args.setup:
    if not args.source_table:
      print("Error: --source-table required for setup mode")
      return 1

    kb_table = (
        args.kb_table or f"{args.source_table.split('.')[0]}.kb_embeddings"
    )

    sql = generate_setup_sql(
        project_id=args.project_id,
        source_table=args.source_table,
        kb_table=kb_table,
        content_column=args.content_column,
        id_column=args.id_column,
        chunk_size=args.chunk_size,
    )

    print("-- RAG Pipeline Setup SQL")
    print(sql)

    if args.dry_run:
      print("\n-- Dry run mode: SQL not executed")
      return 0

    try:
      from google.cloud import bigquery

      client = bigquery.Client(project=args.project_id)

      print("\nExecuting setup (this may take several minutes)...")
      for statement in sql.split(";"):
        statement = statement.strip()
        if statement and not statement.startswith("--"):
          print(f"  Running: {statement[:60]}...")
          client.query(statement + ";").result()

      print("\nSetup complete!")
      return 0

    except ImportError:
      print("\n-- Run in BigQuery Console to execute")
      return 0

  # Handle query mode
  if not args.query:
    print("Error: --query required (or use --setup for pipeline setup)")
    return 1

  if not args.kb_table:
    print("Error: --kb-table required for query mode")
    return 1

  # Determine models
  dataset = args.kb_table.split(".")[0]
  embedding_model = (
      args.embedding_model or f"{args.project_id}.{dataset}.rag_embedding_model"
  )
  generation_model = (
      args.generation_model
      or f"{args.project_id}.{dataset}.rag_generation_model"
  )

  sql = generate_rag_query_sql(
      project_id=args.project_id,
      kb_table=f"{args.project_id}.{args.kb_table}",
      embedding_model=embedding_model,
      generation_model=generation_model,
      query_text=args.query,
      num_sources=args.num_sources,
      max_output_tokens=args.max_tokens,
      temperature=args.temperature,
      include_sources=not args.no_sources,
  )

  if args.output == "sql" or args.dry_run:
    print(sql)
    if args.dry_run:
      print("\n-- Dry run mode: SQL not executed")
    return 0

  # Execute query
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=args.project_id)

    print(f"Querying: {args.query[:60]}...")
    result = list(client.query(sql).result())[0]

    if args.output == "json":
      output = {
          "question": result.question,
          "answer": result.answer,
      }
      if hasattr(result, "sources") and result.sources:
        output["sources"] = [
            {
                "chunk_id": s["chunk_id"],
                "excerpt": s["excerpt"],
                "relevance": s["relevance"],
            }
            for s in result.sources
        ]
      print(json.dumps(output, indent=2))
    else:
      print(f"\nQuestion: {result.question}")
      print(f"\nAnswer: {result.answer}")

      if hasattr(result, "sources") and result.sources:
        print("\nSources:")
        for i, source in enumerate(result.sources, 1):
          print(f"  [{i}] (relevance: {source['relevance']:.2f})")
          print(f"      {source['excerpt'][:100]}...")

    return 0

  except ImportError:
    print("\n-- google-cloud-bigquery not installed")
    print("-- Run: pip install google-cloud-bigquery")
    print("-- Or execute SQL in BigQuery Console")
    return 0

  except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
