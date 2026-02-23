---
name: csv-aggregation
description: Aggregate and summarize CSV data by computing statistics on specified columns.
---

# CSV Aggregation Skill

Analyze CSV data by computing aggregate statistics (sum, mean, min, max, count) grouped by a specified column.

## Available Scripts

### `aggregate.py`

Reads CSV data from stdin and computes aggregate statistics.

**Usage**: `execute_skill_script(skill_name="csv-aggregation", script_name="aggregate.py", input_args="group_col=department metric_col=salary")`

The script expects CSV data piped via stdin or provided as the `data` argument. Pass column names as arguments:
- `group_col`: The column to group by
- `metric_col`: The column to compute statistics on

**Output format**:
```
Group: <group_name>
  count: <n>
  sum: <total>
  mean: <average>
  min: <minimum>
  max: <maximum>
```

## References

- [sample-data.md](./references/sample-data.md) — Sample CSV dataset for testing

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `load_skill_resource` to load the sample data reference.
3. Use `execute_skill_script` with the appropriate arguments to aggregate the data.
4. Present the aggregated results to the user.
