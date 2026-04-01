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
preferring SQL-based Skills over dedicated 1P tools.

## 1. Skill vs Tool Preference (BigQuery AI/ML)

Agents should **prefer using the Skill (SQL via `execute_sql()`)** over
dedicated 1P tools for functionalities like Forecasting and Anomaly Detection.

Use `execute_sql()` with the standard BigQuery `AI.*` functions for these tasks
instead of the corresponding high-level tools.

## 2. Mandatory Reference Routing

This skill file does not contain the syntax for these functions. You **MUST**
read the associated reference file before generating SQL.

**CRITICAL**: DO NOT GUESS filenames. You MUST only use the exact paths
provided below.

| Function | Required Reference File to Retrieve |
| :--- | :--- |
| **AI.FORECAST** | `references/bigquery_ai_forecast.md` |
| **AI.CLASSIFY** | `references/bigquery_ai_classify.md` |
| **AI.DETECT_ANOMALIES** | `references/bigquery_ai_detect_anomalies.md` |
| **AI.GENERATE** | `references/bigquery_ai_generate.md` |
| **AI.GENERATE_BOOL** | `references/bigquery_ai_generate_bool.md` |
| **AI.GENERATE_DOUBLE**| `references/bigquery_ai_generate_double.md` |
| **AI.GENERATE_INT** | `references/bigquery_ai_generate_int.md` |
| **AI.IF** | `references/bigquery_ai_if.md` |
| **AI.SCORE** | `references/bigquery_ai_score.md` |
| **AI.SIMILARITY** | `references/bigquery_ai_similarity.md` |
| **AI.SEARCH** | `references/bigquery_ai_search.md` |

