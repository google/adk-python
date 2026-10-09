---
name: bigquery-ai-ml
license: Apache-2.0
metadata:
  author: google-adk
  version: "1.0"
description: >-
  Leverages BigQuery's built-in machine learning and GenAI capabilities
  for advanced data analytics. Use when you need to write SQL queries
  that perform time-series forecasting, predict values, detect outliers or anomalies, find key drivers,
  perform semantic search or vector search, classify text, calculate similarity,
  summarize content, translate language, evaluate models, filter by semantic conditions,
  measure the causal effect of an intervention, compute correlations between columns,
  detect change points or structural breaks, extract trend or seasonality components,
  or leverage generative AI capabilities in BigQuery. Do not use for general
  BigQuery dataset, table, or job management requests.
---

# BigQuery AI & ML

BigQuery integrates with Vertex AI to provide powerful machine learning and
generative AI capabilities directly within SQL queries using built-in functions
like `AI.FORECAST`, `AI.KEY_DRIVERS`, `AI.DETECT_ANOMALIES`, and `AI.GENERATE`.

## Reference Directory

-   **Functions Reference**:

    -   **AI.AGG**: [ai_agg.md](references/ai_agg.md) - Multi-row semantic
        aggregation and summarization.
    -   **AI.CAUSAL_EFFECT**:
        [ai_causal_effect.md](references/ai_causal_effect.md) - Quantifies the
        impact of an intervention on a time series.
    -   **AI.CLASSIFY**: [ai_classify.md](references/ai_classify.md) - Classify
        text.
    -   **AI.DETECT_ANOMALIES**:
        [ai_detect_anomalies.md](references/ai_detect_anomalies.md) - Detect
        anomalies.
    -   **AI.EVALUATE**: [ai_evaluate.md](references/ai_evaluate.md) - Evaluate
        models.
    -   **AI.FORECAST**: [ai_forecast.md](references/ai_forecast.md) -
        Time-series forecasting.
    -   **AI.GENERATE**: [ai_generate.md](references/ai_generate.md) - Generate
        text using LLMs.
    -   **AI.GENERATE_EMBEDDING**:
        [ai_generate_embedding.md](references/ai_generate_embedding.md) -
        Generate embeddings.
    -   **AI.GENERATE_TABLE**:
        [ai_generate_table.md](references/ai_generate_table.md) - Table-valued
        AI generation.
    -   **AI.IF**: [ai_if.md](references/ai_if.md) - Evaluate semantic
        conditions.
    -   **AI.KEY_DRIVERS**: [ai_key_drivers.md](references/ai_key_drivers.md) -
        Identifies key drivers, this is a TVF.
    -   **AI.SCORE**: [ai_score.md](references/ai_score.md) - Score data.
    -   **AI.SEARCH**: [ai_search.md](references/ai_search.md) - Semantic
        search.
    -   **AI.SIMILARITY**: [ai_similarity.md](references/ai_similarity.md) -
        Semantic similarity.
    -   **Remote Models**: [remote_models.md](references/remote_models.md) -
        Working with remote models (Vertex AI).
    -   **CONTRIBUTION_ANALYSIS**:
        [ml_contribution_analysis.md](references/ml_contribution_analysis.md)
        -   Finds contributing factors, key drivers of change. Requires creating
            a MODEL entity.
    -   **ML.CORRELATION**: [ml_correlation.md](references/ml_correlation.md) -
        Calculates correlation between columns, optionally sliced by dimensions.
    -   **ML.DETECT_CHANGE_POINTS**:
        [ml_detect_change_points.md](references/ml_detect_change_points.md) -
        Detects structural breaks or sustained shifts in a time series.
    -   **ML.SEASONALITY**: [ml_seasonality.md](references/ml_seasonality.md) -
        Extracts seasonal components from a time series.
    -   **ML.TREND**: [ml_trend.md](references/ml_trend.md) - Extracts the
        long-term trend component from a time series.
    -   **VECTOR_SEARCH**: [vector_search.md](references/vector_search.md) -
        Vector search best practices.
