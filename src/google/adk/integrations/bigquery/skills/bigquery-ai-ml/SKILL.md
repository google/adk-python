---
name: bigquery-ai-ml
license: Apache-2.0
metadata:
  author: google-adk
  version: "1.0"
description: |
  Skill for BigQuery AI and Machine Learning queries using standard SQL
  and `AI.*` functions (preferred over dedicated tools).

---

# Skill: bigquery-ai-ml

This skill defines the usage and rules for BigQuery AI/ML functions,
preferring SQL-based Skills over dedicated BigQuery tools.

## 1. Skill vs Tool Preference (BigQuery AI/ML)

Agents should **prefer using the Skill (SQL via `execute_sql()`)** over
dedicated BigQuery tools for functionalities like Forecasting and Anomaly
Detection.

Use `execute_sql()` with the standard BigQuery `AI.*` functions for these tasks
instead of the corresponding high-level tools.

## 2. Mandatory Reference Routing

This skill file does not contain the syntax for these functions. You **MUST**
read the associated reference file before generating SQL.

**CRITICAL**: DO NOT GUESS filenames. You MUST only use the exact paths
provided below.

| Function | Description | Required Reference File to Retrieve |
| :--- | :--- | :--- |
| **AI.FORECAST** | Time-series forecasting via the pre-trained TimesFM model | `references/bigquery_ai_forecast.md` |
| **AI.CLASSIFY** | Categorize unstructured data into predefined labels | `references/bigquery_ai_classify.md` |
| **AI.DETECT_ANOMALIES** | Identify deviations in time-series data via the pre-trained TimesFM model | `references/bigquery_ai_detect_anomalies.md` |
| **AI.GENERATE** | General-purpose text and content generation | `references/bigquery_ai_generate.md` |
| **AI.GENERATE_BOOL** | Generate a boolean value (TRUE/FALSE) based on a prompt | `references/bigquery_ai_generate_bool.md` |
| **AI.GENERATE_DOUBLE** | Generate a floating-point number based on a prompt | `references/bigquery_ai_generate_double.md` |
| **AI.GENERATE_INT** | Generate an integer value based on a prompt | `references/bigquery_ai_generate_int.md` |
| **AI.IF** | Evaluate a natural-language boolean condition | `references/bigquery_ai_if.md` |
| **AI.SCORE** | Rank items by semantic relevance (use with ORDER BY) | `references/bigquery_ai_score.md` |
| **AI.SIMILARITY** | Compute cosine similarity between two inputs | `references/bigquery_ai_similarity.md` |
| **AI.SEARCH** | Semantic search on tables with autonomous embedding generation | `references/bigquery_ai_search.md` |
