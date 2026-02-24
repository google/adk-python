# BigQueryBench: Skill Invocation & Instruction Adherence Evaluation

## Overview

BigQueryBench evaluates agents built with ADK's `SkillToolset` on
two dimensions:

1. **Skill invocation correctness** (trace-based) — Did the agent
   call the right skill tools with the right arguments?
2. **Instruction adherence** (LLM-as-judge) — Did the agent follow
   the skill's instructions and produce correct results?

The trace-based checks are deterministic. The instruction adherence
checks use natural-language rubrics evaluated by a judge LLM, making
them easy to write and immune to exact-wording variance.

## Quick Start

```bash
# Skill-only mode (no BigQuery credentials needed):
export GOOGLE_CLOUD_API_KEY=your-vertex-ai-api-key

# Full mode (BigQuery + skills — requires ADC + project):
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_API_KEY=your-vertex-ai-api-key

# Run all eval cases
python -m benchmarks.bigquerybench.runner

# Run one case
python -m benchmarks.bigquerybench.runner --filter skill_load

# Dry-run (validate JSON only, no LLM calls)
python -m benchmarks.bigquerybench.runner --dry-run

# Run unit tests (no API keys needed)
pytest tests/unittests/benchmarks/bigquerybench/ -v
```

## How It Works

```
eval_sets/bigquerybench_eval.json
  ↓  (user query + expected tool_uses + rubrics)
runner.py
  ↓  runs agent via ADK Runner
  ↓  collects event trace → Invocations
metrics.py
  ├── tool_invocation_score: expected skill tool names ⊆ actual?
  ├── tool_args_score: expected (tool, skill-arg) pairs ⊆ actual?
  └── instruction_adherence_score: LLM judge checks rubrics
  ↓
PASS if all three scores meet thresholds
```

## Three Metrics

| Metric | Type | What It Checks | Pass Condition |
|--------|------|----------------|----------------|
| `tool_invocation_score` | Trace | Correct skill tools called | Score = 1.0 |
| `tool_args_score` | Trace | Correct skill/resource/script targeted | Score = 1.0 |
| `instruction_adherence_score` | LLM judge | Agent followed instructions, output correct | Score >= 0.75 |

A case **passes** when all three metrics meet their thresholds.

## Eval Case Format

Each eval case has three parts:
1. **`conversation`** — user query + expected skill tool calls
2. **`rubrics`** — natural-language assertions checked by the judge

```json
{
  "eval_id": "skill_load_reference",
  "conversation": [
    {
      "invocation_id": "inv-01",
      "user_content": {
        "parts": [{"text": "Load the public datasets reference from bq-sql-analyst."}],
        "role": "user"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
          {"name": "load_skill_resource", "args": {
            "skill_name": "bq-sql-analyst",
            "path": "references/public-datasets.md"
          }}
        ],
        "tool_responses": [],
        "intermediate_responses": []
      },
      "creation_timestamp": 0.0
    }
  ],
  "rubrics": [
    {
      "rubric_id": "shows_datasets",
      "rubric_content": {
        "text_property": "The response contains information about BigQuery public datasets."
      }
    },
    {
      "rubric_id": "loaded_skill_first",
      "rubric_content": {
        "text_property": "The agent loaded the skill instructions before loading the resource."
      }
    }
  ],
  "creation_timestamp": 0.0
}
```

### Trace Checks (deterministic)

| Field in `args` | Checked? | Why |
|-----------------|----------|-----|
| `name` | Yes | Must load the right skill (`load_skill`) |
| `skill_name` | Yes | Must target the right skill (`load_skill_resource`, `run_skill_script`) |
| `path` | Yes | Must load the right resource (`load_skill_resource`) |
| `script_path` | Yes | Must run the right script (`run_skill_script`) |

### Rubrics (LLM-as-judge)

Each rubric is a natural-language assertion about the agent's behavior
or output. The judge LLM reads the conversation (user request + tool
trace + final response) and answers yes/no per rubric.

**Example rubrics:**
```json
{"rubric_id": "r1", "rubric_content": {"text_property": "The agent used AI.classify to classify the data."}}
{"rubric_id": "r2", "rubric_content": {"text_property": "The result contains a markdown table with group statistics."}}
{"rubric_id": "r3", "rubric_content": {"text_property": "The agent loaded the skill before running the script."}}
```

## Included Eval Cases

| eval_id | Expected Trace | Rubrics |
|---------|---------------|---------|
| `skill_list_skills` | `list_skills()` | Lists bq-sql-analyst; includes description |
| `skill_load_sql_analyst` | `load_skill(name=bq-sql-analyst)` | Describes capabilities; mentions scripts |
| `skill_load_reference` | `load_skill` → `load_skill_resource` | Shows datasets; loaded skill first |
| `skill_query_with_reference` | `load_skill` → `load_skill_resource` | Consulted reference; summarizes datasets; followed workflow |
| `skill_run_format_script` | `load_skill` → `run_skill_script` | Loaded before run; has table; has columns |

## Example Output

```
========================================================================
  BigQueryBench — Skill Evaluation
========================================================================

[1/5] skill_list_skills
    -> list_skills()
  tools=1.00  args=1.00  adherence=1.00  PASS

[2/5] skill_load_sql_analyst
    -> load_skill(name='bq-sql-analyst')
  tools=1.00  args=1.00  adherence=1.00  PASS

[3/5] skill_load_reference
    -> load_skill(name='bq-sql-analyst')
    -> load_skill_resource(skill_name='bq-sql-analyst', path='references/public-datasets.md')
  tools=1.00  args=1.00  adherence=1.00  PASS

[4/5] skill_query_with_reference
    -> load_skill(name='bq-sql-analyst')
    -> load_skill_resource(skill_name='bq-sql-analyst', path='references/public-datasets.md')
  tools=1.00  args=1.00  adherence=1.00  PASS

[5/5] skill_run_format_script
    -> load_skill(name='bq-sql-analyst')
    -> run_skill_script(skill_name='bq-sql-analyst', script_path='scripts/format_results.py')
  tools=1.00  args=1.00  adherence=1.00  PASS

Eval Case                          Tools   Args  Adhere  Result
------------------------------------------------------------------------
skill_list_skills                   1.00   1.00    1.00   PASS
skill_load_sql_analyst              1.00   1.00    1.00   PASS
skill_load_reference                1.00   1.00    1.00   PASS
skill_query_with_reference          1.00   1.00    1.00   PASS
skill_run_format_script             1.00   1.00    1.00   PASS
------------------------------------------------------------------------

========================================================================
  Summary
========================================================================
  Cases:              5/5 (100.0%)
  Avg Tool Match:     1.00
  Avg Args Match:     1.00
  Avg Adherence:      1.00
  Elapsed:            488.8s
========================================================================
```

## Adding a New Eval Case

### For an existing skill

Add a JSON object to `bigquerybench_eval.json` with both `tool_uses`
(trace expectations) and `rubrics` (instruction adherence assertions).

```json
{
  "eval_id": "skill_explore_usa_names",
  "conversation": [
    {
      "invocation_id": "inv-new-01",
      "user_content": {
        "parts": [{"text": "Use the bq-sql-analyst skill to explore the usa_names dataset."}],
        "role": "user"
      },
      "intermediate_data": {
        "tool_uses": [
          {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
          {"name": "load_skill_resource", "args": {"skill_name": "bq-sql-analyst", "path": "references/public-datasets.md"}}
        ],
        "tool_responses": [],
        "intermediate_responses": []
      },
      "creation_timestamp": 0.0
    }
  ],
  "rubrics": [
    {
      "rubric_id": "consulted_ref",
      "rubric_content": {"text_property": "The agent consulted the public datasets reference."}
    },
    {
      "rubric_id": "mentions_usa_names",
      "rubric_content": {"text_property": "The response mentions the usa_names dataset and its columns."}
    }
  ],
  "creation_timestamp": 0.0
}
```

### For a new skill

1. Create a skill directory under `skills/`:
   ```
   skills/my-new-skill/
   ├── SKILL.md
   ├── references/
   └── scripts/
   ```

2. Register it in `agent.py`:
   ```python
   _SKILL_NAMES = [
       "bq-sql-analyst",
       "my-new-skill",  # ← add here
   ]
   ```

3. Add eval cases with trace expectations + rubrics.

### Writing Good Rubrics

**Do:**
- Assert observable behavior: "The agent loaded the skill before running the script."
- Assert output properties: "The response contains a table with columns name and count."
- Assert domain correctness: "The result includes the top 3 Shakespeare works."

**Don't:**
- Assert exact wording: "The response starts with 'Here are the results'."
- Assert implementation details: "The agent called execute_sql with SELECT DISTINCT."
- Use vague assertions: "The response is good."

### When Do You Need Code Changes?

| Scenario | JSON | `metrics.py` | `runner.py` | `agent.py` |
|----------|:----:|:------------:|:-----------:|:----------:|
| New eval case, existing skill | Yes | - | - | - |
| New skill added to `skills/` | Yes | - | - | Yes (add to `_SKILL_NAMES`) |
| Change judge model or threshold | - | Yes | - | - |
| Need entirely new metric | Yes | Yes | Yes | - |
| Agent instruction change | - | - | - | Yes |

## Architecture

```
benchmarks/bigquerybench/
├── __init__.py
├── agent.py           # LlmAgent + BigQueryToolset + SkillToolset
├── runner.py          # Runs agent, scores trace + rubrics
├── metrics.py         # 3 metrics: trace (x2) + LLM-as-judge (x1)
├── eval_sets/
│   └── bigquerybench_eval.json   # 5 eval cases with rubrics
└── skills/
    └── bq-sql-analyst/
        ├── SKILL.md
        ├── references/
        │   └── public-datasets.md
        └── scripts/
            └── format_results.py

tests/unittests/benchmarks/bigquerybench/
└── test_metrics.py    # 14 tests (trace + LLM judge + JSON validation)
```

## Retry Backoff

Both the agent model and the LLM judge use exponential backoff with
retry on 429 (rate limit) errors:

- **Agent model**: 5 attempts, 2s initial delay, 2x exponential
  backoff (via `HttpRetryOptions`)
- **LLM judge**: 5 attempts, 2s → 4s → 8s → 16s manual backoff,
  plus 3 HTTP-level retries per attempt

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLOUD_API_KEY` | Yes | Vertex AI API key for agent model + judge |
| `GOOGLE_CLOUD_PROJECT` | No | GCP project for BigQuery API (enables BigQuery toolset) |
| `BQ_EVAL_WRITE_MODE` | No | `blocked` (default) / `protected` / `allowed` |

**Two modes:**
- **Skill-only** (default): Set `GOOGLE_CLOUD_API_KEY` only.
  BigQuery toolset is skipped; all 5 skill eval cases run.
- **Full mode**: Set both `GOOGLE_CLOUD_API_KEY` and
  `GOOGLE_CLOUD_PROJECT` (+ ADC configured). BigQuery toolset
  is enabled alongside skills.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `tool_invocation_score = 0` | Agent didn't call expected skill tool — check agent instructions |
| `tool_args_score < 1.0` | Agent targeted wrong skill or resource — check user query specificity |
| `adherence < 0.75` | Agent produced wrong output — review rubrics and skill instructions |
| 429 RESOURCE_EXHAUSTED | Rate limit — retry backoff handles this automatically; wait and retry |
| Skill not found | Verify skill dir exists in `skills/` and name is in `_SKILL_NAMES` in `agent.py` |
| Judge LLM fails | Check `GOOGLE_CLOUD_API_KEY` is set correctly |
| `load_skill_resource` fails | Check the `path` arg matches a real file under the skill dir |
