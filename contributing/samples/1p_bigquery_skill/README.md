# 1P BigQuery Skill Sample

This sample demonstrates the **1P (first-party) Skills** pattern: combining
a raw toolset (`BigQueryToolset`) with a curated skill (`SkillToolset`) to
give the agent both tools and guided workflows.

## What This Shows

- **BigQueryToolset** provides raw tools: `execute_sql`, `list_dataset_ids`,
  `get_table_info`, etc.
- **SkillToolset** provides skill discovery and loading: `list_skills`,
  `load_skill`, `load_skill_resource`, `run_skill_script`.
- The **bigquery-data-analysis** skill ships pre-packaged with ADK and
  follows the [agentskills.io specification](https://agentskills.io/specification).

The agent can discover the skill at runtime, load its instructions for
guided workflows, and access reference materials on-demand.

## Setup

1. Install ADK with BigQuery extras:

   ```bash
   pip install google-adk[bigquery]
   ```

2. Set up OAuth credentials:

   - Create OAuth 2.0 credentials in the Google Cloud Console.
   - Update `agent.py` with your `client_id` and `client_secret`.

3. Run the sample:

   ```bash
   adk web contributing/samples
   ```

4. Select `1p_bigquery_skill` from the agent list.

## How It Works

```
User Query
    |
    v
LlmAgent (gemini-2.5-flash)
    |
    +-- BigQueryToolset tools (direct data access)
    |     list_dataset_ids, list_table_ids, get_table_info,
    |     execute_sql, forecast, detect_anomalies, ...
    |
    +-- SkillToolset tools (guided workflows)
          list_skills         -> discovers "bigquery-data-analysis"
          load_skill          -> loads step-by-step instructions
          load_skill_resource -> loads sql_patterns.md, etc.
          run_skill_script    -> executes skill scripts
```

## Progressive Disclosure

The skill uses three levels of content:

1. **L1 - Metadata** (always available): skill name and description shown
   via `list_skills`.
2. **L2 - Instructions** (on activation): full workflow steps loaded via
   `load_skill`.
3. **L3 - References** (on demand): detailed SQL patterns, schema
   exploration guides, and error handling loaded via `load_skill_resource`.

This keeps the agent's context efficient while making deep knowledge
available when needed.

## Extending This Pattern

Other toolsets can follow the same pattern:

1. Create a spec-compliant skill directory under `integration/<toolset>/skills/`.
2. Add a `get_*_skill()` convenience loader.
3. (Optional) Add an alias in `tools/<toolset>/` for backward compatibility.
4. Users add both the toolset and `SkillToolset` to their agent's tools.
