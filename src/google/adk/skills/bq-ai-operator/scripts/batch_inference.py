#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate SQL for batch inference jobs in BigQuery."""

import argparse
import json
import sys
from typing import Optional


def generate_text_inference_sql(
    model_name: str,
    source_table: str,
    prompt_column: str,
    output_table: str,
    id_column: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    batch_size: Optional[int] = None,
) -> str:
    """Generate SQL for batch text generation.

    Args:
        model_name: Full model name (project.dataset.model)
        source_table: Source table with input data
        prompt_column: Column containing prompts
        output_table: Destination table for results
        id_column: Optional ID column to preserve
        temperature: LLM temperature
        max_output_tokens: Max output tokens
        batch_size: Optional batch size limit

    Returns:
        SQL statement for batch inference
    """
    id_select = f"{id_column}," if id_column else ""
    batch_clause = f"LIMIT {batch_size}" if batch_size else ""

    return f"""-- Batch text generation inference
CREATE OR REPLACE TABLE `{output_table}` AS
SELECT
  {id_select}
  {prompt_column} AS input_prompt,
  ml_generate_text_result['candidates'][0]['content']['parts'][0]['text'] AS generated_text,
  ml_generate_text_result['usageMetadata']['promptTokenCount'] AS prompt_tokens,
  ml_generate_text_result['usageMetadata']['candidatesTokenCount'] AS output_tokens,
  ml_generate_text_status AS status,
  CURRENT_TIMESTAMP() AS inference_timestamp
FROM ML.GENERATE_TEXT(
  MODEL `{model_name}`,
  (
    SELECT {id_select} {prompt_column}
    FROM `{source_table}`
    WHERE {prompt_column} IS NOT NULL
    {batch_clause}
  ),
  STRUCT(
    {temperature} AS temperature,
    {max_output_tokens} AS max_output_tokens,
    TRUE AS flatten_json_output
  )
);

-- Check results
SELECT
  COUNT(*) AS total_rows,
  COUNTIF(status = '') AS successful,
  COUNTIF(status != '') AS failed,
  SUM(prompt_tokens) AS total_prompt_tokens,
  SUM(output_tokens) AS total_output_tokens
FROM `{output_table}`;
"""


def generate_embedding_inference_sql(
    model_name: str,
    source_table: str,
    content_column: str,
    output_table: str,
    id_column: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> str:
    """Generate SQL for batch embedding generation.

    Args:
        model_name: Full model name (project.dataset.model)
        source_table: Source table with input data
        content_column: Column containing text to embed
        output_table: Destination table for results
        id_column: Optional ID column to preserve
        batch_size: Optional batch size limit

    Returns:
        SQL statement for batch embedding inference
    """
    id_select = f"{id_column}," if id_column else ""
    batch_clause = f"LIMIT {batch_size}" if batch_size else ""

    return f"""-- Batch embedding generation
CREATE OR REPLACE TABLE `{output_table}` AS
SELECT
  {id_select}
  {content_column} AS content,
  ml_generate_embedding_result['embeddings'][0]['values'] AS embedding,
  ml_generate_embedding_result['embeddings'][0]['statistics']['token_count'] AS token_count,
  ml_generate_embedding_status AS status,
  CURRENT_TIMESTAMP() AS inference_timestamp
FROM ML.GENERATE_EMBEDDING(
  MODEL `{model_name}`,
  (
    SELECT {id_select} {content_column}
    FROM `{source_table}`
    WHERE {content_column} IS NOT NULL
      AND LENGTH({content_column}) > 0
    {batch_clause}
  ),
  STRUCT(TRUE AS flatten_json_output)
);

-- Check results
SELECT
  COUNT(*) AS total_rows,
  COUNTIF(status = '') AS successful,
  COUNTIF(status != '') AS failed,
  AVG(ARRAY_LENGTH(embedding)) AS avg_embedding_dim
FROM `{output_table}`;
"""


def generate_incremental_sql(
    model_name: str,
    source_table: str,
    content_column: str,
    output_table: str,
    id_column: str,
    inference_type: str = "embedding",
) -> str:
    """Generate SQL for incremental inference (only new rows).

    Args:
        model_name: Full model name
        source_table: Source table
        content_column: Content column
        output_table: Output table (must exist)
        id_column: ID column for tracking
        inference_type: Type of inference (embedding, text)

    Returns:
        SQL statement for incremental inference
    """
    if inference_type == "embedding":
        return f"""-- Incremental embedding generation (new rows only)
INSERT INTO `{output_table}`
SELECT
  {id_column},
  {content_column} AS content,
  ml_generate_embedding_result['embeddings'][0]['values'] AS embedding,
  ml_generate_embedding_result['embeddings'][0]['statistics']['token_count'] AS token_count,
  ml_generate_embedding_status AS status,
  CURRENT_TIMESTAMP() AS inference_timestamp
FROM ML.GENERATE_EMBEDDING(
  MODEL `{model_name}`,
  (
    SELECT s.{id_column}, s.{content_column}
    FROM `{source_table}` s
    LEFT JOIN `{output_table}` o ON s.{id_column} = o.{id_column}
    WHERE o.{id_column} IS NULL
      AND s.{content_column} IS NOT NULL
  ),
  STRUCT(TRUE AS flatten_json_output)
);
"""
    else:
        return f"""-- Incremental text generation (new rows only)
INSERT INTO `{output_table}`
SELECT
  {id_column},
  {content_column} AS input_prompt,
  ml_generate_text_result['candidates'][0]['content']['parts'][0]['text'] AS generated_text,
  ml_generate_text_status AS status,
  CURRENT_TIMESTAMP() AS inference_timestamp
FROM ML.GENERATE_TEXT(
  MODEL `{model_name}`,
  (
    SELECT s.{id_column}, s.{content_column}
    FROM `{source_table}` s
    LEFT JOIN `{output_table}` o ON s.{id_column} = o.{id_column}
    WHERE o.{id_column} IS NULL
      AND s.{content_column} IS NOT NULL
  ),
  STRUCT(0.2 AS temperature, 1024 AS max_output_tokens)
);
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL for batch inference jobs"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Full model name (project.dataset.model)",
    )
    parser.add_argument(
        "--source-table",
        type=str,
        required=True,
        help="Source table with input data",
    )
    parser.add_argument(
        "--content-column",
        type=str,
        required=True,
        help="Column containing content/prompts",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        required=True,
        help="Destination table for results",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        help="ID column to preserve",
    )
    parser.add_argument(
        "--inference-type",
        type=str,
        choices=["embedding", "text"],
        default="embedding",
        help="Type of inference",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Generate incremental inference SQL",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Limit batch size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for text generation",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Max output tokens for text generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["sql", "json"],
        default="sql",
        help="Output format",
    )

    args = parser.parse_args()

    if args.incremental:
        if not args.id_column:
            print("Error: --id-column required for incremental mode", file=sys.stderr)
            sys.exit(1)
        sql = generate_incremental_sql(
            args.model_name,
            args.source_table,
            args.content_column,
            args.output_table,
            args.id_column,
            args.inference_type,
        )
    elif args.inference_type == "embedding":
        sql = generate_embedding_inference_sql(
            args.model_name,
            args.source_table,
            args.content_column,
            args.output_table,
            args.id_column,
            args.batch_size,
        )
    else:
        sql = generate_text_inference_sql(
            args.model_name,
            args.source_table,
            args.content_column,
            args.output_table,
            args.id_column,
            args.temperature,
            args.max_output_tokens,
            args.batch_size,
        )

    if args.output == "json":
        result = {
            "sql": sql,
            "model_name": args.model_name,
            "source_table": args.source_table,
            "output_table": args.output_table,
            "inference_type": args.inference_type,
            "incremental": args.incremental,
        }
        print(json.dumps(result, indent=2))
    else:
        print(sql)


if __name__ == "__main__":
    main()
