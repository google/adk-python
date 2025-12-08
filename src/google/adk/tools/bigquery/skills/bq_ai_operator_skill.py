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

"""BigQuery AI Operator Skill for ADK.

This skill provides AI/ML functions for generative AI operations directly
in BigQuery SQL, including:

- AI.GENERATE: Flexible text/multimodal generation
- AI.CLASSIFY: Categorize data into user-defined classes
- AI.EXTRACT: Extract structured data from unstructured text
- AI.IF: Filter data based on natural language conditions
- AI.SCORE: Rate/rank data by quality or relevance
- AI.EMBED: Generate embeddings for semantic search
- AI.SIMILARITY: Compare embeddings for clustering/recommendations

For full documentation, see:
https://cloud.google.com/bigquery/docs/generative-ai-overview
"""

from __future__ import annotations

from typing import Any
from typing import List

from pydantic import Field

from ....skills.base_skill import BaseSkill
from ....skills.base_skill import SkillConfig
from ....utils.feature_decorator import experimental


@experimental
class BQAIOperatorSkill(BaseSkill):
  """Skill for BigQuery AI Operator functions.

  This skill bundles AI functions for generative AI operations in BigQuery,
  allowing agents to use LLMs directly within SQL queries.

  Key capabilities:
  - Text generation and summarization
  - Classification into custom categories
  - Entity/data extraction from unstructured text
  - Semantic filtering with natural language
  - Quality scoring and ranking
  - Embedding generation for vector search
  - Similarity comparison for recommendations

  Example:
    ```python
    from google.adk.tools.bigquery.skills import BQAIOperatorSkill

    skill = BQAIOperatorSkill()
    agent = Agent(
        name="ai_analyst",
        skills=[skill],
        enable_programmatic_tool_calling=True,
    )
    ```

  Note: Requires BigQuery Enterprise edition and appropriate permissions.
  """

  name: str = Field(default="bq_ai_operator")
  description: str = Field(
      default=(
          "AI-powered analysis in BigQuery using generative AI. "
          "Generate text, classify data, extract entities, filter with "
          "natural language, score/rank content, create embeddings, and "
          "compute semantic similarity - all directly in SQL."
      )
  )
  config: SkillConfig = Field(
      default_factory=lambda: SkillConfig(
          timeout_seconds=180.0,  # AI operations can take time
          max_parallel_calls=10,
      )
  )

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    """Return tool declarations for BQ AI Operator functions."""
    return [
        {
            "name": "ai_generate",
            "description": (
                "Generate text using AI.GENERATE. The most flexible inference "
                "function - analyze any combination of text and unstructured "
                "data. Use for summarization, translation, Q&A, content "
                "generation, and more."
            ),
            "parameters": {
                "input_data": "SQL query or table with input data",
                "prompt": "The prompt template with column references",
                "model": "Optional model name (default: gemini-pro)",
                "output_column": "Name for the output column",
            },
        },
        {
            "name": "ai_generate_table",
            "description": (
                "Generate structured table output using AI.GENERATE_TABLE. "
                "Extracts structured data into a table with a custom schema "
                "from unstructured input."
            ),
            "parameters": {
                "input_data": "SQL query or table with input data",
                "prompt": "The prompt describing what to extract",
                "output_schema": "Schema for output table columns",
                "model": "Optional model name",
            },
        },
        {
            "name": "ai_classify",
            "description": (
                "Classify data into user-defined categories using AI.CLASSIFY. "
                "Groups text/multimodal data into predefined classes."
            ),
            "parameters": {
                "input_data": "SQL query or table with data to classify",
                "input_column": "Column containing text to classify",
                "categories": "List of category names/labels",
                "output_column": "Name for the classification result column",
            },
        },
        {
            "name": "ai_extract",
            "description": (
                "Extract structured information from unstructured text using "
                "AI. Useful for entity extraction, parsing documents, etc."
            ),
            "parameters": {
                "input_data": "SQL query or table with unstructured text",
                "input_column": "Column containing text to process",
                "extraction_spec": "What to extract (entities, fields, etc.)",
                "output_columns": "Names for extracted data columns",
            },
        },
        {
            "name": "ai_if",
            "description": (
                "Filter data using natural language conditions with AI.IF. "
                "Returns TRUE/FALSE based on whether rows match the condition."
            ),
            "parameters": {
                "input_data": "SQL query or table to filter",
                "input_column": "Column to evaluate",
                "condition": "Natural language condition to check",
                "output_column": "Name for the boolean result column",
            },
        },
        {
            "name": "ai_score",
            "description": (
                "Rate/score data using AI.SCORE. Assigns numeric scores to "
                "rank rows by quality, relevance, or other criteria."
            ),
            "parameters": {
                "input_data": "SQL query or table to score",
                "input_column": "Column containing content to score",
                "criteria": "Scoring criteria in natural language",
                "output_column": "Name for the score column",
            },
        },
        {
            "name": "ai_embed",
            "description": (
                "Generate embeddings using AI.EMBED or AI.GENERATE_EMBEDDING. "
                "Creates high-dimensional vectors for semantic search, "
                "clustering, and similarity comparisons."
            ),
            "parameters": {
                "input_data": "SQL query or table with text to embed",
                "input_column": "Column containing text",
                "model": "Embedding model (default: text-embedding-004)",
                "output_column": "Name for the embedding vector column",
            },
        },
        {
            "name": "ai_similarity",
            "description": (
                "Compare embeddings using AI.SIMILARITY. Computes cosine "
                "similarity between embedding vectors for clustering, "
                "recommendations, and duplicate detection."
            ),
            "parameters": {
                "embedding1": "First embedding column or value",
                "embedding2": "Second embedding column or value",
                "output_column": "Name for the similarity score column",
            },
        },
        {
            "name": "ai_forecast",
            "description": (
                "Generate time series forecasts using AI.FORECAST. "
                "Uses TimesFM model for point and interval predictions."
            ),
            "parameters": {
                "input_data": "SQL query or table with time series data",
                "timestamp_col": "Column containing timestamps",
                "data_col": "Column containing values to forecast",
                "horizon": "Number of time steps to forecast",
                "id_cols": "Optional columns identifying multiple series",
            },
        },
    ]

  def get_orchestration_template(self) -> str:
    """Return example orchestration code for BQ AI Operator functions."""
    return '''
async def analyze_customer_feedback(tools):
    """Analyze customer feedback using AI functions."""
    import asyncio

    # Classify sentiment and extract entities in parallel
    classify_result, extract_result = await asyncio.gather(
        tools.ai_classify(
            input_data="""
                SELECT feedback_id, feedback_text
                FROM `my_project.my_dataset.customer_feedback`
                WHERE feedback_date >= '2024-01-01'
            """,
            input_column="feedback_text",
            categories=["positive", "negative", "neutral", "mixed"],
            output_column="sentiment"
        ),
        tools.ai_extract(
            input_data="""
                SELECT feedback_id, feedback_text
                FROM `my_project.my_dataset.customer_feedback`
                WHERE feedback_date >= '2024-01-01'
            """,
            input_column="feedback_text",
            extraction_spec="product names, feature requests, complaints",
            output_columns=["products", "feature_requests", "complaints"]
        )
    )

    # Summarize the results
    sentiment_counts = {}
    for row in classify_result.get("rows", []):
        sentiment = row.get("sentiment")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    return {
        "total_feedback": len(classify_result.get("rows", [])),
        "sentiment_distribution": sentiment_counts,
        "sample_extractions": extract_result.get("rows", [])[:10]
    }


async def semantic_search(tools):
    """Perform semantic search using embeddings."""

    # Generate embeddings for search query
    query_embedding = await tools.ai_embed(
        input_data="SELECT 'How do I reset my password?' as query",
        input_column="query",
        model="text-embedding-004",
        output_column="query_embedding"
    )

    # Find similar documents
    similar_docs = await tools.ai_similarity(
        embedding1="query_embedding",
        embedding2="doc_embedding",
        output_column="similarity_score"
    )

    # Filter to most relevant results
    relevant = [
        row for row in similar_docs.get("rows", [])
        if row.get("similarity_score", 0) > 0.7
    ]

    return {
        "query": "How do I reset my password?",
        "num_results": len(relevant),
        "top_matches": relevant[:5]
    }


async def content_moderation(tools):
    """Filter inappropriate content using AI.IF."""

    # Filter content based on natural language criteria
    filtered = await tools.ai_if(
        input_data="""
            SELECT post_id, content, author_id
            FROM `my_project.my_dataset.user_posts`
            WHERE created_at >= CURRENT_DATE() - 7
        """,
        input_column="content",
        condition="content is appropriate for all ages and does not contain spam, harassment, or explicit material",
        output_column="is_appropriate"
    )

    # Score remaining content for quality
    scored = await tools.ai_score(
        input_data="""
            SELECT post_id, content
            FROM appropriate_posts
        """,
        input_column="content",
        criteria="helpfulness, clarity, and engagement potential on a scale of 1-10",
        output_column="quality_score"
    )

    appropriate_count = sum(
        1 for row in filtered.get("rows", [])
        if row.get("is_appropriate")
    )

    return {
        "total_posts": len(filtered.get("rows", [])),
        "appropriate_posts": appropriate_count,
        "flagged_posts": len(filtered.get("rows", [])) - appropriate_count,
        "top_quality_posts": sorted(
            scored.get("rows", []),
            key=lambda x: x.get("quality_score", 0),
            reverse=True
        )[:10]
    }
'''

  def filter_result(self, result: Any) -> Any:
    """Filter BQ AI Operator results to reduce context size.

    - Truncates large result sets
    - Truncates very long generated text
    - Summarizes embedding vectors (too large to return in full)
    """
    if not isinstance(result, dict):
      return result

    filtered = result.copy()

    # Truncate large row results
    if "rows" in filtered and len(filtered["rows"]) > 50:
      filtered["rows"] = filtered["rows"][:50]
      filtered["result_truncated"] = True
      filtered["total_rows"] = len(result["rows"])

    # Process individual rows
    if "rows" in filtered:
      for row in filtered["rows"]:
        if isinstance(row, dict):
          for key, value in row.items():
            # Truncate very long text outputs
            if isinstance(value, str) and len(value) > 2000:
              row[key] = value[:2000] + "... [truncated]"

            # Summarize embedding vectors (don't send full vector)
            if isinstance(value, list) and len(value) > 100:
              if all(isinstance(x, (int, float)) for x in value[:10]):
                row[key] = {
                    "type": "embedding_vector",
                    "dimensions": len(value),
                    "sample": value[:5],
                    "note": "Full vector truncated for context efficiency",
                }

            # Round numeric values
            if isinstance(value, float):
              row[key] = round(value, 4)

    return filtered
