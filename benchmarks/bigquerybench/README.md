# BigQueryBench: End-to-End Evaluation for BigQuery Skills

## Overview

BigQueryBench is a reusable evaluation pipeline for any skill or agent
built with ADK's `BigQueryToolset`. It mirrors the SkillsBench
architecture (`benchmarks/skillsbench/`) but targets BigQuery-specific
tool chains: schema exploration, SQL generation, AI/ML operations
(forecast, anomaly detection, contribution analysis), and multi-step
analytical workflows.

**Design goals:**

- **Reusable:** Add a new BigQuery skill by dropping one eval case
  JSON and one optional reference SQL file — no code changes needed.
- **Reproducible:** All eval cases use BigQuery public datasets
  (`bigquery-public-data`) so any GCP project with BigQuery API
  enabled can run the suite.
- **Layered metrics:** Three dimensions scored independently —
  schema discovery, tool-call coverage, output correctness.
- **CI-friendly:** Single `python -m benchmarks.bigquerybench.runner`
  invocation with JSON results and exit code.

## Quick Start

```bash
# Prerequisites
# 1. GCP project with BigQuery API enabled
# 2. Application Default Credentials configured:
#    gcloud auth application-default login
# 3. ADK installed with BigQuery extras:
#    uv sync --all-extras

# Set environment
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_GENAI_USE_VERTEXAI=1  # or use GOOGLE_API_KEY

# Run all eval cases
python -m benchmarks.bigquerybench.runner

# Run specific eval case(s)
python -m benchmarks.bigquerybench.runner --filter sql_public_dataset

# Run with multiple attempts for variance measurement
python -m benchmarks.bigquerybench.runner --num-runs 3

# Dry-run mode (validates eval set JSON, no LLM calls)
python -m benchmarks.bigquerybench.runner --dry-run
```

## Architecture

```
benchmarks/bigquerybench/
├── README.md                          # This file
├── __init__.py
├── agent.py                           # Root agent with BigQueryToolset
├── runner.py                          # Standalone evaluation runner
├── metrics.py                         # BigQuery-specific custom metrics
└── eval_sets/
    └── bigquerybench_eval.json        # Eval cases (public datasets)
```

### Relationship to SkillsBench

| Aspect | SkillsBench | BigQueryBench |
|--------|-------------|---------------|
| Toolset | `SkillToolset` (4 tools) | `BigQueryToolset` (10 tools) |
| Discovery tools | `list_skills`, `load_skill` | `list_dataset_ids`, `list_table_ids`, `get_dataset_info`, `get_table_info` |
| Execution tools | `run_skill_script` | `execute_sql`, `forecast`, `detect_anomalies`, `analyze_contribution` |
| Data source | Bundled reference files | BigQuery public datasets |
| Auth | None (local files) | GCP credentials (ADC / service account / OAuth) |
| Metrics | discovery, tool_usage, binary | schema_discovery, tool_usage, output_correctness |

## Evaluation Pipeline

### Stage 1: Agent Setup

The agent under test is defined in `agent.py`. It uses
`BigQueryToolset` with read-only defaults:

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.bigquery.bigquery_credentials import (
    BigQueryCredentialsConfig,
)
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
import google.auth

credentials, _ = google.auth.default()
credentials_config = BigQueryCredentialsConfig(credentials=credentials)

tool_config = BigQueryToolConfig(
    write_mode=WriteMode.BLOCKED,   # Read-only for eval safety
    max_query_result_rows=50,
)

bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config,
    bigquery_tool_config=tool_config,
)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="bigquerybench_agent",
    description="Agent for BigQuery data exploration and analysis.",
    instruction="""\
        You are a data analyst with access to BigQuery tools.
        Use them to explore schemas, run SQL queries, and answer
        the user's questions about data. Always explore the schema
        (list datasets, list tables, get table info) before writing
        SQL. Show query results clearly.
    """,
    tools=[bigquery_toolset],
)
```

### Stage 2: Eval Set Definition

Each eval case is a JSON object in `eval_sets/bigquerybench_eval.json`
following the ADK `EvalSet` schema. The key fields are:

```json
{
  "eval_id": "unique_case_id",
  "conversation": [
    {
      "invocation_id": "inv-01",
      "user_content": {
        "parts": [{"text": "User's question about BigQuery data"}],
        "role": "user"
      },
      "final_response": {
        "parts": [{"text": "Expected key phrases in the answer"}],
        "role": "model"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "list_dataset_ids", "args": {"project_id": "bigquery-public-data"}},
          {"name": "get_table_info", "args": {"project_id": "...", "dataset_id": "...", "table_id": "..."}},
          {"name": "execute_sql", "args": {"project_id": "...", "query": "SELECT ..."}}
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

**Key conventions:**

- `final_response.parts[0].text` contains **reference lines** — key
  phrases that must appear in the agent's response for a pass. One
  phrase per line. Case-insensitive substring match.
- `intermediate_data.tool_uses` lists the **expected tool call
  sequence**. The `tool_usage` metric checks set coverage (not
  strict ordering). The `schema_discovery` metric checks that at
  least one schema-exploration tool was called.
- All eval cases use `bigquery-public-data` project datasets so
  results are deterministic and reproducible.

### Stage 3: Metrics

Three custom metrics in `metrics.py`, following the ADK custom metric
function signature:

#### 3a. `schema_discovery_score`

Checks whether the agent explored the schema before querying. Scores
1.0 if any of these tools were called: `list_dataset_ids`,
`list_table_ids`, `get_dataset_info`, `get_table_info`. Scores 0.0
otherwise.

**Rationale:** Agents that skip schema exploration and guess table
names produce fragile SQL that breaks on schema changes. This metric
enforces the "explore before query" pattern.

#### 3b. `tool_usage_score`

Fraction of expected tool calls actually made:
`|expected_tools ∩ actual_tools| / |expected_tools|`.

Uses set-based matching (any order). Passes at threshold >= 0.5.

Same semantics as SkillsBench `tool_usage_score`, reused for
consistency.

#### 3c. `output_correctness_score`

Binary pass/fail: 1.0 if the agent's final response contains all
expected reference lines (case-insensitive substring match). 0.0
otherwise.

Same semantics as SkillsBench `skillsbench_binary_score`, reused for
consistency.

### Stage 4: Runner Execution

```
runner.py
    ↓
Load agent from benchmarks/bigquerybench/agent.py
    ↓
Load eval set from eval_sets/bigquerybench_eval.json
    ↓
For each eval case:
    ↓
    Run agent.run_async(user_query) via ADK Runner
        ↓ (generates events)
    Convert events → Invocation (with intermediate_data.tool_uses)
        ↓
    Apply schema_discovery_score
    Apply tool_usage_score
    Apply output_correctness_score
        ↓
    Record per-case scores
    ↓
Aggregate scores → leaderboard summary
    ↓
Print table + exit code (0 = all pass, 1 = any fail)
```

## Eval Case Catalog

The following eval cases are included. They are organized by
complexity tier to test progressively harder agent capabilities.

### Tier 1: Schema Exploration

These test the agent's ability to navigate BigQuery metadata.

| eval_id | Dataset | User Query | Expected Tools |
|---------|---------|-----------|----------------|
| `schema_list_datasets` | `bigquery-public-data` | "What datasets are available in bigquery-public-data?" | `list_dataset_ids` |
| `schema_list_tables` | `usa_names` | "What tables exist in the usa_names dataset?" | `list_dataset_ids` → `list_table_ids` |
| `schema_get_table_info` | `usa_names.usa_1910_current` | "What columns and types does the usa_1910_current table have?" | `list_table_ids` → `get_table_info` |

### Tier 2: SQL Generation & Execution

These test SQL generation against public data with known answers.

| eval_id | Dataset | User Query | Expected Tools | Reference Output |
|---------|---------|-----------|----------------|-----------------|
| `sql_top_names` | `usa_names` | "What are the top 5 most popular baby names in 2020?" | `get_table_info` → `execute_sql` | Top names by count |
| `sql_aggregation` | `usa_names` | "How many distinct names were registered each decade from 1950 to 2000?" | `get_table_info` → `execute_sql` | Decade counts |
| `sql_public_dataset` | `samples.shakespeare` | "Which Shakespeare work has the most unique words?" | `get_table_info` → `execute_sql` | Work name + count |

### Tier 3: Multi-Step Analysis

These test the agent's ability to chain multiple tools.

| eval_id | Dataset | User Query | Expected Tools | Reference Output |
|---------|---------|-----------|----------------|-----------------|
| `multi_step_explore_and_query` | `austin_bikeshare` | "Explore the Austin bikeshare dataset and tell me the top 5 busiest stations by trip count." | `list_table_ids` → `get_table_info` → `execute_sql` | Station names + counts |

## Adding a New BigQuery Eval Case

To add a new eval case (e.g., for a new BigQuery AI operator skill):

### Step 1: Identify the public dataset

Pick a dataset from `bigquery-public-data` that exercises the skill.
Verify it exists:

```sql
SELECT * FROM `bigquery-public-data.DATASET.INFORMATION_SCHEMA.TABLES`
LIMIT 5;
```

### Step 2: Write the eval case JSON

Add a new object to the `eval_cases` array in
`eval_sets/bigquerybench_eval.json`:

```json
{
  "eval_id": "your_unique_eval_id",
  "conversation": [
    {
      "invocation_id": "inv-your-id-01",
      "user_content": {
        "parts": [{"text": "Your user query here"}],
        "role": "user"
      },
      "final_response": {
        "parts": [{"text": "reference line 1\nreference line 2"}],
        "role": "model"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "tool_name", "args": {"arg1": "val1"}}
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

### Step 3: Choose reference lines

Run the expected query manually and pick 3-5 key phrases from the
result that are **stable** (won't change if data is appended). Good
reference lines:

- Column names or schema facts ("column: name, type: STRING")
- Aggregation results from historical data ("hamlet", "king lear")
- Structural facts ("3 tables", "5 columns")

Avoid: exact row counts on append-only tables, floating-point values
that may shift with precision.

### Step 4: Validate

```bash
# Run just your new case
python -m benchmarks.bigquerybench.runner --filter your_unique_eval_id
```

### Step 5: Commit

Add only the modified `bigquerybench_eval.json`. No code changes
needed.

## Eval Case Template for AI/ML Tools

For `forecast`, `detect_anomalies`, and `analyze_contribution` skills,
use this template:

```json
{
  "eval_id": "forecast_weather_temperature",
  "conversation": [
    {
      "invocation_id": "inv-forecast-01",
      "user_content": {
        "parts": [{"text": "Forecast the next 7 days of average temperature using the NOAA GSOD weather data for station 725300 (Chicago O'Hare) from 2023."}],
        "role": "user"
      },
      "final_response": {
        "parts": [{"text": "forecast_timestamp\nforecast_value\nprediction_interval"}],
        "role": "model"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "get_table_info", "args": {"project_id": "bigquery-public-data", "dataset_id": "noaa_gsod", "table_id": "gsod2023"}},
          {"name": "forecast", "args": {"project_id": "your-project", "history_data": "SELECT date, temp FROM `bigquery-public-data.noaa_gsod.gsod2023` WHERE stn = '725300'", "timestamp_col": "date", "data_col": "temp", "horizon": 7}}
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

**Key considerations for AI/ML eval cases:**

- `forecast`, `analyze_contribution`, and `detect_anomalies` create
  temporary BigQuery ML models. Ensure `write_mode` is at least
  `PROTECTED` (anonymous dataset) or `ALLOWED`.
- Reference lines should validate **structural output** (column names
  like `forecast_timestamp`, `is_anomaly`) rather than exact numeric
  values, since ML model outputs vary across runs.
- Set `tool_config.write_mode = WriteMode.PROTECTED` in the agent
  for AI/ML eval cases that need to create temp models.

## Complete Walkthrough: Adding a New BigQuery Skill

This section walks through every step of updating the evaluation
pipeline when a **new BigQuery tool** is added to
`BigQueryToolset`. We use a concrete example: a hypothetical
`cluster_data` tool that performs K-Means clustering via BQML.

### Context: What Is the New Tool?

Suppose a developer adds this tool to `src/google/adk/tools/bigquery/`:

```python
def cluster_data(
    project_id: str,
    input_data: str,           # Table ID or SQL query
    feature_cols: list[str],   # Columns to cluster on
    num_clusters: int = 3,     # K in K-Means
    *,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    tool_context: ToolContext,
) -> dict:
    """Cluster rows using BigQuery ML K-Means.

    Creates a TEMP MODEL and returns cluster assignments
    with centroid distances.
    """
```

The tool generates SQL like:

```sql
CREATE TEMP MODEL cluster_model_<uuid>
  OPTIONS (MODEL_TYPE='KMEANS', NUM_CLUSTERS=3)
  AS SELECT feature1, feature2 FROM `project.dataset.table`;

SELECT * FROM ML.PREDICT(MODEL cluster_model_<uuid>,
  (SELECT feature1, feature2 FROM `project.dataset.table`));
```

Output columns: `centroid_id`, `nearest_centroids_distance`,
plus the original feature columns.

### Step 1: Register the Tool in BigQueryToolset

The developer registers the tool in `bigquery_toolset.py`. Once
registered, it's automatically available to any agent using
`BigQueryToolset`. **No changes to the eval agent** (`agent.py`)
are needed — the toolset dynamically exposes all registered tools.

Verify the tool is visible:

```python
from benchmarks.bigquerybench.agent import bigquery_toolset
tools = bigquery_toolset.get_tools()
assert any(t.name == "cluster_data" for t in tools)
```

### Step 2: Decide If Existing Metrics Are Sufficient

Check each metric against the new tool's behavior:

| Metric | Does It Work? | Action Needed? |
|--------|--------------|----------------|
| `schema_discovery_score` | Yes — the agent should still explore schema before clustering | No change |
| `tool_usage_score` | Yes — set-based matching works for any tool name | No change |
| `output_correctness_score` | **Partially** — ML outputs vary across runs, so exact numeric matching will be flaky | Use structural reference lines (column names, cluster count) instead of exact values |

**When you DO need a new metric:** If the new tool has a unique
correctness criterion that can't be captured by substring matching
(e.g., "the SQL must be syntactically valid", "the forecast horizon
must match the request"), add a new metric function to `metrics.py`.
See [Step 2b: Adding a Custom Metric](#step-2b-adding-a-custom-metric)
below.

### Step 3: Pick a Public Dataset

Choose a dataset from `bigquery-public-data` with numeric columns
suitable for clustering. For this example, we'll use the **penguins**
dataset (`ml_datasets.penguins`) which has well-known numeric features.

Verify it exists:

```sql
SELECT column_name, data_type
FROM `bigquery-public-data.ml_datasets.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'penguins';
```

Expected columns: `species`, `island`, `culmen_length_mm`,
`culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`.

### Step 4: Write the Eval Case

Add to `eval_sets/bigquerybench_eval.json`:

```json
{
  "eval_id": "ml_cluster_penguins",
  "conversation": [
    {
      "invocation_id": "inv-cluster-01",
      "user_content": {
        "parts": [{"text": "Cluster the penguins in bigquery-public-data.ml_datasets.penguins into 3 groups based on their physical measurements (culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g). Show the cluster assignments."}],
        "role": "user"
      },
      "final_response": {
        "parts": [{"text": "centroid_id\nculmen_length_mm\nculmen_depth_mm\nflipper_length_mm\nbody_mass_g"}],
        "role": "model"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "get_table_info", "args": {"project_id": "bigquery-public-data", "dataset_id": "ml_datasets", "table_id": "penguins"}},
          {"name": "cluster_data", "args": {"project_id": "your-project", "input_data": "SELECT culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g FROM `bigquery-public-data.ml_datasets.penguins` WHERE body_mass_g IS NOT NULL", "feature_cols": ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"], "num_clusters": 3}}
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

**Reference line design choices:**

- `centroid_id` — structural: confirms clustering output format
- `culmen_length_mm`, `culmen_depth_mm`, etc. — structural: confirms
  feature columns are returned in output
- We do NOT include exact centroid values or row counts because ML
  model outputs vary across runs

### Step 5: Update the Agent Write Mode

The `cluster_data` tool creates a `TEMP MODEL`, which requires at
least `PROTECTED` write mode. Update the eval run command:

```bash
BQ_EVAL_WRITE_MODE=protected \
  python -m benchmarks.bigquerybench.runner --filter ml_cluster_penguins
```

Or for CI, set it in the environment configuration.

### Step 6: Validate the Eval Case

```bash
# Dry-run: check JSON is valid
python -m benchmarks.bigquerybench.runner --dry-run

# Single-case run
BQ_EVAL_WRITE_MODE=protected \
  python -m benchmarks.bigquerybench.runner --filter ml_cluster_penguins

# Multi-run for variance check (ML outputs may vary)
BQ_EVAL_WRITE_MODE=protected \
  python -m benchmarks.bigquerybench.runner --filter ml_cluster --num-runs 3
```

Expected output:

```
=================================================================
  BigQueryBench Evaluation — ADK BigQueryToolset
=================================================================

[1/1] Running: ml_cluster_penguins
  Response: Here are the cluster assignments for the penguins...
  Scores: schema=1.0 tools=1.00 output=PASS

Eval Case                                  Schema   Tools   Output   Result
---------------------------------------------------------------------------
ml_cluster_penguins                           1.0    1.00      1.0    PASS
---------------------------------------------------------------------------

=================================================================
  Leaderboard Summary
=================================================================
  Framework:          ADK BigQueryToolset
  Cases:              1/1 (100.0%)
  Avg Schema Disc.:   1.00
  Avg Tool Usage:     1.00
  Elapsed:            12.3s
=================================================================
```

### Step 7: Commit

```bash
git add benchmarks/bigquerybench/eval_sets/bigquerybench_eval.json
git commit -m "eval(bigquerybench): add clustering eval case for cluster_data tool"
```

**Files changed:** Only `bigquerybench_eval.json`. No code changes
needed unless a new metric was added (Step 2b).

---

### Step 2b: Adding a Custom Metric (When Needed)

If existing metrics are insufficient for your new tool, add a metric
function to `metrics.py`. Here's a concrete example: a
`clustering_quality_score` that validates the agent used the correct
number of clusters.

**1. Write the metric function in `metrics.py`:**

```python
def clustering_quality_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Score 1.0 if cluster_data was called with correct num_clusters.

  Checks that the agent called cluster_data and that the
  num_clusters argument matches the expected value.
  """
  if not expected_invocations:
    return EvaluationResult(
        overall_score=1.0,
        overall_eval_status=EvalStatus.PASSED,
    )

  # Extract expected num_clusters from expected tool calls.
  expected_k = None
  for inv in expected_invocations:
    for tc in get_all_tool_calls(inv.intermediate_data):
      if tc.name == "cluster_data" and tc.args:
        expected_k = tc.args.get("num_clusters")
        break

  # Extract actual num_clusters from actual tool calls.
  actual_k = None
  for inv in actual_invocations:
    for tc in get_all_tool_calls(inv.intermediate_data):
      if tc.name == "cluster_data" and tc.args:
        actual_k = tc.args.get("num_clusters")
        break

  if expected_k is None:
    score = 1.0  # No clustering expected
  elif actual_k is None:
    score = 0.0  # Clustering expected but not called
  else:
    score = 1.0 if actual_k == expected_k else 0.0

  status = (
      EvalStatus.PASSED if score >= 1.0
      else EvalStatus.FAILED
  )
  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
      per_invocation_results=_make_per_invocation(
          actual_invocations, expected_invocations,
          score, status,
      ),
  )
```

**2. Wire it into the runner (`runner.py`):**

```python
from .metrics import clustering_quality_score

def score_invocations(actual, expected):
    # ... existing metrics ...

    result = clustering_quality_score(
        metric, actual_invocations, expected_invocations,
    )
    scores["clustering_quality"] = result.overall_score or 0.0

    return scores
```

**3. Update the results table to show the new column.**

**4. Commit all changed files:**

```bash
git add benchmarks/bigquerybench/metrics.py \
        benchmarks/bigquerybench/runner.py \
        benchmarks/bigquerybench/eval_sets/bigquerybench_eval.json
git commit -m "eval(bigquerybench): add cluster_data eval case and clustering_quality metric"
```

---

### Summary: Files to Touch per Scenario

| Scenario | `eval.json` | `metrics.py` | `runner.py` | `agent.py` |
|----------|:-----------:|:------------:|:-----------:|:----------:|
| New eval case for existing tool | **Yes** | No | No | No |
| New tool, existing metrics sufficient | **Yes** | No | No | No |
| New tool, needs custom metric | **Yes** | **Yes** | **Yes** | No |
| New tool, needs agent config change | **Yes** | Maybe | Maybe | **Yes** |

The toolset auto-discovers new tools, so `agent.py` only changes if
the agent's instruction prompt or write mode needs updating.

## Metrics Reference

| Metric | Function Path | Threshold | Pass Condition |
|--------|--------------|-----------|----------------|
| Schema Discovery | `benchmarks.bigquerybench.metrics.schema_discovery_score` | 1.0 | Any schema tool called |
| Tool Usage | `benchmarks.bigquerybench.metrics.tool_usage_score` | 0.5 | >= 50% expected tools called |
| Output Correctness | `benchmarks.bigquerybench.metrics.output_correctness_score` | 1.0 | All reference lines present |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project for BigQuery API calls |
| `GOOGLE_GENAI_USE_VERTEXAI` | Conditional | Set to `1` for Vertex AI LLM backend |
| `GOOGLE_API_KEY` | Conditional | API key for Google AI Studio backend |
| `BQ_EVAL_WRITE_MODE` | No | Override write mode (`blocked`/`protected`/`allowed`). Default: `blocked` |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `403 Access Denied` | Missing BigQuery API access | Enable BigQuery API in GCP console; run `gcloud auth application-default login` |
| `execute_sql` returns empty | Query references wrong project | Ensure public dataset queries use `bigquery-public-data` as project |
| `forecast` fails with write error | `write_mode=BLOCKED` | Set `BQ_EVAL_WRITE_MODE=protected` for AI/ML eval cases |
| Low `schema_discovery_score` | Agent skips exploration | Strengthen agent instructions to always explore schema first |
| Flaky `output_correctness_score` | Reference lines too specific | Use structural phrases, not exact numeric values |
