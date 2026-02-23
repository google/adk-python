---
name: json-transform
description: Transform JSON data by flattening nested structures and renaming fields.
---

# JSON Transform Skill

Transform JSON objects by flattening nested structures, renaming keys, and filtering fields.

## Available Scripts

### `transform.py`

Transforms a JSON object according to a field mapping specification.

**Usage**: `execute_skill_script(skill_name="json-transform", script_name="transform.py", input_args="flatten=true")`

The script reads the embedded sample data and applies transformations:
- `flatten=true`: Flatten nested objects using dot notation
- `rename=old:new`: Rename a field (can be specified multiple times)
- `keep=field1,field2`: Keep only specified fields

**Output format**: Pretty-printed JSON

## References

- [sample-data.md](./references/sample-data.md) — Sample nested JSON for testing

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `load_skill_resource` to examine the sample data.
3. Use `execute_skill_script` to transform the data.
4. Present the transformed JSON to the user.
