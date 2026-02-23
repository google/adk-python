---
name: log-parsing
description: Parse and analyze structured log files to extract error patterns and statistics.
metadata:
  category: system-admin
  aliases: log-analyzer,syslog-parser
---

# Log Parsing Skill

Parse structured log files to extract error counts, warning summaries, and time-based patterns. Supports common log formats.

## Available Scripts

### `parse.py`

Analyzes embedded sample log data and produces a summary report.

**Usage**: `execute_skill_script(skill_name="log-parsing", script_name="parse.py", input_args="level=ERROR")`

Arguments:
- `level`: Filter by log level (DEBUG, INFO, WARNING, ERROR, ALL). Default: ALL
- `format`: Output format (summary, detail, timeline). Default: summary

**Output format** (summary):
```
Log Analysis Report
===================
Total entries: <n>
ERROR: <count>
WARNING: <count>
INFO: <count>
DEBUG: <count>
```

**Output format** (detail):
Lists each matching log entry with timestamp and message.

## References

- [sample-logs.md](./references/sample-logs.md) — Sample log data for testing

## Workflow

1. Use `load_skill` to read these instructions.
2. Optionally use `load_skill_resource` to examine the sample log data.
3. Use `execute_skill_script` with the desired log level filter.
4. Present the analysis report to the user.
