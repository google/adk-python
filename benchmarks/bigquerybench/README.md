# BigQueryBench: Trace-Based Evaluation for BigQuery Skills

## Overview

BigQueryBench verifies that an agent built with ADK's `BigQueryToolset`
**calls the correct tools with the correct arguments**.  It inspects
the tool-call trace only — no response text matching.

For each eval case, the pipeline checks two things:

1. **Tool invocation** — Did the agent call the right BigQuery
   functions?  (e.g., `get_table_info` then `execute_sql`)
2. **Tool arguments** — Did those calls point at the right data?
   (e.g., `project_id="bigquery-public-data"`,
   `dataset_id="usa_names"`, `table_id="usa_1910_current"`)

This makes evaluation deterministic, easy to maintain, and immune to
LLM response wording variance.

## Quick Start

```bash
# Prerequisites: GCP project with BigQuery API + ADC configured
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_GENAI_USE_VERTEXAI=1

# Run all eval cases
python -m benchmarks.bigquerybench.runner

# Run one case
python -m benchmarks.bigquerybench.runner --filter schema_list_tables

# Dry-run (validate JSON only, no LLM calls)
python -m benchmarks.bigquerybench.runner --dry-run
```

## How It Works

```
eval_sets/bigquerybench_eval.json
  ↓  (user query + expected tool_uses)
runner.py
  ↓  runs agent via ADK Runner
  ↓  collects event trace → Invocations
metrics.py
  ├── tool_invocation_score: expected tool names ⊆ actual tool names?
  └── tool_args_score: expected (tool, project/dataset/table) ⊆ actual?
  ↓
PASS if both scores = 1.0
```

## Eval Case Format

Each eval case specifies a user query and the expected tool calls.
No `final_response` is needed — only the trace matters.

```json
{
  "eval_id": "schema_get_table_info",
  "conversation": [
    {
      "invocation_id": "inv-01",
      "user_content": {
        "parts": [{"text": "What columns does usa_1910_current have?"}],
        "role": "user"
      },
      "intermediate_data": {
        "tool_uses": [
          {
            "name": "get_table_info",
            "args": {
              "project_id": "bigquery-public-data",
              "dataset_id": "usa_names",
              "table_id": "usa_1910_current"
            }
          }
        ],
        "tool_responses": [],
        "intermediate_responses": []
      },
      "creation_timestamp": 0.0
    }
  ],
  "creation_timestamp": 0.0
}
```

**What gets checked:**

| Field in `args` | Checked? | Why |
|-----------------|----------|-----|
| `project_id` | Yes | Must point at the right GCP project |
| `dataset_id` | Yes | Must load the right dataset |
| `table_id` | Yes | Must load the right table |
| `query` | **No** | Exact SQL varies — agent may write equivalent SQL differently |
| Other args | **No** | Tool-specific args (e.g., `horizon`, `num_clusters`) are not checked by default |

## Metrics

| Metric | What It Checks | Pass Condition |
|--------|---------------|----------------|
| `tool_invocation_score` | All expected tool names appear in the trace | Score = 1.0 |
| `tool_args_score` | All expected `(tool, project_id/dataset_id/table_id)` pairs appear in the trace | Score = 1.0 |

A case **passes** when both scores are 1.0.

## Included Eval Cases

| eval_id | User Query | Expected Trace |
|---------|-----------|----------------|
| `schema_list_datasets` | "What datasets are in bigquery-public-data?" | `list_dataset_ids(project_id=bigquery-public-data)` |
| `schema_list_tables` | "What tables in usa_names?" | `list_table_ids(project_id=.., dataset_id=usa_names)` |
| `schema_get_table_info` | "Columns of usa_1910_current?" | `get_table_info(project_id=.., dataset_id=usa_names, table_id=usa_1910_current)` |
| `sql_shakespeare_unique_words` | "Top 3 works by unique words?" | `get_table_info(.., shakespeare)` → `execute_sql(..)` |
| `sql_usa_names_top_2020` | "Top 5 baby names in 2020?" | `get_table_info(.., usa_1910_current)` → `execute_sql(..)` |
| `sql_names_by_decade` | "Distinct names per decade 1950-2000?" | `get_table_info(.., usa_1910_current)` → `execute_sql(..)` |
| `multi_step_explore_and_query` | "Explore bikeshare, top 5 stations?" | `list_table_ids(..)` → `get_table_info(.., bikeshare_trips)` → `execute_sql(..)` |

## Adding a New Eval Case

### For an existing tool (e.g., new `execute_sql` scenario)

Only add a JSON object to `bigquerybench_eval.json`. No code changes.

```json
{
  "eval_id": "sql_weather_hottest_day",
  "conversation": [
    {
      "invocation_id": "inv-weather-01",
      "user_content": {
        "parts": [{"text": "What was the hottest day recorded in the NOAA GSOD 2023 data?"}],
        "role": "user"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "get_table_info", "args": {"project_id": "bigquery-public-data", "dataset_id": "noaa_gsod", "table_id": "gsod2023"}},
          {"name": "execute_sql", "args": {"project_id": "bigquery-public-data"}}
        ],
        "tool_responses": [],
        "intermediate_responses": []
      },
      "creation_timestamp": 0.0
    }
  ],
  "creation_timestamp": 0.0
}
```

Validate: `python -m benchmarks.bigquerybench.runner --filter sql_weather`

### For a new tool (e.g., `forecast`, `cluster_data`)

Same steps — just use the new tool name in `tool_uses`. The metrics
check tool names and key args generically, so no metric code changes
are needed.

```json
{
  "eval_id": "ml_forecast_temperature",
  "conversation": [
    {
      "invocation_id": "inv-forecast-01",
      "user_content": {
        "parts": [{"text": "Forecast the next 7 days of temperature from NOAA GSOD 2023 data for station 725300."}],
        "role": "user"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "get_table_info", "args": {"project_id": "bigquery-public-data", "dataset_id": "noaa_gsod", "table_id": "gsod2023"}},
          {"name": "forecast", "args": {"project_id": "bigquery-public-data"}}
        ],
        "tool_responses": [],
        "intermediate_responses": []
      },
      "creation_timestamp": 0.0
    }
  ],
  "creation_timestamp": 0.0
}
```

For AI/ML tools that create temp models, set write mode:
```bash
BQ_EVAL_WRITE_MODE=protected python -m benchmarks.bigquerybench.runner --filter ml_forecast
```

## Complete Walkthrough: Adding a New BigQuery Skill

This walkthrough uses a concrete example: a hypothetical
`cluster_data` tool (K-Means via BQML) being added to
`BigQueryToolset`.

### Step 1: Register the tool

The developer adds `cluster_data` to `bigquery_toolset.py`. Once
registered, it's automatically available to the eval agent — no
changes to `agent.py` needed.

### Step 2: Write the eval case

What we want to verify: when the user asks "cluster the penguins
data", the agent should:
1. Call `get_table_info` on `ml_datasets.penguins` (load the schema)
2. Call `cluster_data` against `bigquery-public-data` (invoke the
   right tool on the right data)

```json
{
  "eval_id": "ml_cluster_penguins",
  "conversation": [
    {
      "invocation_id": "inv-cluster-01",
      "user_content": {
        "parts": [{"text": "Cluster the penguins in bigquery-public-data.ml_datasets.penguins into 3 groups based on their physical measurements."}],
        "role": "user"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "get_table_info", "args": {"project_id": "bigquery-public-data", "dataset_id": "ml_datasets", "table_id": "penguins"}},
          {"name": "cluster_data", "args": {"project_id": "bigquery-public-data"}}
        ],
        "tool_responses": [],
        "intermediate_responses": []
      },
      "creation_timestamp": 0.0
    }
  ],
  "creation_timestamp": 0.0
}
```

**What gets checked automatically:**
- `tool_invocation_score`: Did the trace contain both
  `get_table_info` and `cluster_data`?
- `tool_args_score`: Did `get_table_info` target
  `(bigquery-public-data, ml_datasets, penguins)`? Did
  `cluster_data` target `bigquery-public-data`?

**What is NOT checked** (intentionally):
- The exact `feature_cols` or `num_clusters` the LLM chose
- The exact response wording
- The numeric clustering results

### Step 3: Validate

```bash
BQ_EVAL_WRITE_MODE=protected \
  python -m benchmarks.bigquerybench.runner --filter ml_cluster_penguins
```

Expected output:

```
[1/1] ml_cluster_penguins
    -> get_table_info(project_id='bigquery-public-data', dataset_id='ml_datasets', table_id='penguins')
    -> cluster_data(project_id='bigquery-public-data')
  tools=1.00  args=1.00  PASS
```

### Step 4: Commit

```bash
git add benchmarks/bigquerybench/eval_sets/bigquerybench_eval.json
git commit -m "eval(bigquerybench): add clustering eval for cluster_data"
```

**Only `bigquerybench_eval.json` changed.** No code changes.

### When Do You Need Code Changes?

| Scenario | JSON | `metrics.py` | `runner.py` | `agent.py` |
|----------|:----:|:------------:|:-----------:|:----------:|
| New eval case, existing tool | Yes | - | - | - |
| New tool, trace check is enough | Yes | - | - | - |
| New tool, need to check a non-key arg (e.g., `num_clusters`) | Yes | Yes (add arg to `_KEY_ARGS`) | - | - |
| New tool, need entirely new metric | Yes | Yes | Yes | - |
| Agent instruction or write-mode change | - | - | - | Yes |

**Adding a checked arg** is a one-line change in `metrics.py`:

```python
_KEY_ARGS = frozenset({
    "project_id",
    "dataset_id",
    "table_id",
    "num_clusters",  # ← add here
})
```

## Architecture

```
benchmarks/bigquerybench/
├── __init__.py
├── agent.py           # LlmAgent + BigQueryToolset (read-only default)
├── runner.py          # Runs agent, collects trace, scores
├── metrics.py         # tool_invocation_score + tool_args_score
└── eval_sets/
    └── bigquerybench_eval.json   # 7 eval cases
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project for BigQuery API |
| `GOOGLE_GENAI_USE_VERTEXAI` | Conditional | `1` for Vertex AI backend |
| `GOOGLE_API_KEY` | Conditional | API key for AI Studio backend |
| `BQ_EVAL_WRITE_MODE` | No | `blocked` (default) / `protected` / `allowed` |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `403 Access Denied` | `gcloud auth application-default login` + enable BigQuery API |
| `tool_invocation_score = 0` | Agent didn't call expected tool — check agent instructions |
| `tool_args_score < 1.0` | Agent pointed at wrong dataset/table — check user query specificity |
| AI/ML tool fails | Set `BQ_EVAL_WRITE_MODE=protected` |
