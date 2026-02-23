# SkillsBench Evaluation Harness for ADK

Evaluates ADK's `SkillToolset` against tasks adapted from the
[SkillsBench](https://github.com/benchflow-ai/skillsbench) benchmark.

## Overview

This harness adapts 8 representative SkillsBench tasks as ADK skills and
evaluates them through the ADK evaluation framework. It tests whether an
agent can discover, load, and execute skills using the `SkillToolset`
tools: `list_skills`, `load_skill`, `load_skill_resource`, and
`execute_skill_script`.

## Task Categories

| # | Category | Skill | What it tests |
|---|----------|-------|---------------|
| 1 | Data Analysis | csv-aggregation | skill discovery + script execution |
| 2 | File Processing | json-transform | load_skill_resource + script |
| 3 | Web Scraping | html-extraction | skill with references |
| 4 | API Interaction | rest-client | multi-step skill usage |
| 5 | Text Transformation | regex-replace | simple script execution |
| 6 | Code Generation | function-scaffold | skill instruction following |
| 7 | Math Computation | statistical-calc | output validation |
| 8 | System Admin | log-parsing | complex skill with metadata |

## Setup

```bash
# From repo root
uv venv --python "python3.11" ".venv"
source .venv/bin/activate
uv sync --all-extras

# Set your API key
export GOOGLE_API_KEY="your-key-here"
```

## Usage

### Run with ADK CLI

```bash
# Interactive web UI
adk web benchmarks/skillsbench

# Run evaluation via ADK eval
adk eval benchmarks/skillsbench \
    benchmarks/skillsbench/eval_sets/skillsbench_eval.json
```

### Run standalone scorer

```bash
python benchmarks/skillsbench/runner.py
python benchmarks/skillsbench/runner.py --num-runs 3
python benchmarks/skillsbench/runner.py --eval-set path/to/custom_eval.json
```

### Output format

The standalone runner produces a per-task results table and a
leaderboard-format summary:

```
============================================================
  Leaderboard Summary
============================================================
  Model:              gemini-2.5-flash
  Framework:          ADK SkillToolset
  Tasks:              X/8 (XX.X%)
  Avg Discovery:      X.XX
  Avg Tool Usage:     X.XX
  Elapsed:            XX.Xs
============================================================
```

## Custom Metrics

Three metrics are provided in `metrics.py`:

- **skill_discovery_score** — 1.0 if the agent called both `list_skills`
  and `load_skill`, else 0.0
- **tool_usage_score** — Fraction of expected tool calls that were made
  (ANY_ORDER matching)
- **skillsbench_binary_score** — 1.0 if the final response contains all
  expected reference lines, else 0.0

Reference these in eval configs via their dotted paths:
```
benchmarks.skillsbench.metrics.skill_discovery_score
benchmarks.skillsbench.metrics.tool_usage_score
benchmarks.skillsbench.metrics.skillsbench_binary_score
```

## Directory Structure

```
benchmarks/skillsbench/
├── __init__.py
├── README.md
├── agent.py                     # ADK agent with SkillToolset
├── skills/                      # 8 adapted SkillsBench tasks
│   ├── csv-aggregation/
│   ├── json-transform/
│   ├── html-extraction/
│   ├── rest-client/
│   ├── regex-replace/
│   ├── function-scaffold/
│   ├── statistical-calc/
│   └── log-parsing/
├── eval_sets/
│   └── skillsbench_eval.json    # EvalSet with 8 cases
├── metrics.py                   # Custom metric functions
└── runner.py                    # Standalone runner
```

## Adding New Tasks

1. Create a skill directory under `skills/` with a `SKILL.md` following
   the [Agent Skills spec](https://github.com/benchflow-ai/skillsbench)
2. Add scripts under `skills/<name>/scripts/`
3. Add references under `skills/<name>/references/` (optional)
4. Add the skill name to `_SKILL_NAMES` in `agent.py`
5. Add a new `EvalCase` entry to `eval_sets/skillsbench_eval.json`

## Security Note

This harness uses `UnsafeLocalCodeExecutor` for skill script execution.
For production or untrusted skill scripts, use `ContainerCodeExecutor`
or `VertexAICodeExecutor` instead.
