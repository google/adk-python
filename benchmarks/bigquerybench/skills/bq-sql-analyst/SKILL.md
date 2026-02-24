---
name: bq-sql-analyst
description: Analyze BigQuery public datasets with SQL — explore schemas, write queries, and format results.
---

# BigQuery SQL Analyst Skill

Explore and analyze BigQuery public datasets by examining schemas,
writing SQL queries, and presenting formatted results.

## Available Scripts

### `format_results.py`

Formats raw query output as a readable markdown table.

**Usage**: `run_skill_script(skill_name="bq-sql-analyst", script_path="scripts/format_results.py", args={"header": "name,count", "rows": "Alice,5;Bob,3"})`

Arguments:
- `header`: Comma-separated column names
- `rows`: Semicolon-separated rows, each with comma-separated values
- `title` (optional): Table title

## References

- [public-datasets.md](./references/public-datasets.md) — Commonly used BigQuery public datasets and their schemas

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `load_skill_resource` to review the public datasets reference.
3. Use BigQuery tools (`get_table_info`, `execute_sql`) to explore and query data.
4. Optionally use `run_skill_script` to format results.
5. Present findings to the user.
