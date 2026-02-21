---
name: python-helper
description: Python utility scripts for code analysis, data processing, and generation tasks.
version: 1.0.0
---

# Python Helper Skill

A collection of lightweight Python utility scripts for common development tasks.

## Available Scripts

### `fibonacci.py`
Generates a Fibonacci sequence. Pass the desired count as an argument.

**Usage**: `execute_skill_script(skill_name="python-helper", script_name="fibonacci.py", input_args="10")`

### `word_count.py`
Analyzes text and reports word frequency statistics. Pass the text to analyze as an argument.

**Usage**: `execute_skill_script(skill_name="python-helper", script_name="word_count.py", input_args="the quick brown fox jumps over the lazy dog the fox")`

### `json_format.py`
Pretty-prints and validates a JSON string. Pass the JSON as a single quoted argument.

**Usage**: `execute_skill_script(skill_name="python-helper", script_name="json_format.py", input_args='{"name":"Alice","scores":[90,85,92]}')`

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `load_skill_resource` to inspect a script's source if needed.
3. Use `execute_skill_script` with appropriate `input_args` to run a script.
4. Interpret the script's stdout and present results to the user.
