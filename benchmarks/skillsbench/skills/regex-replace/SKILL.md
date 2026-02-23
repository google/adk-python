---
name: regex-replace
description: Perform regex-based find-and-replace operations on text input.
---

# Regex Replace Skill

Apply regular expression patterns to find and replace text. Supports basic and advanced regex syntax.

## Available Scripts

### `replace.py`

Performs regex find-and-replace on input text.

**Usage**: `execute_skill_script(skill_name="regex-replace", script_name="replace.py", input_args="pattern=\\d+ replacement=NUM text='Order 123 has 45 items at $67'")`

Arguments:
- `pattern`: The regex pattern to match
- `replacement`: The replacement string
- `text`: The input text to process
- `count`: Maximum replacements (default: all)

**Output format**:
```
Original: <input text>
Pattern: <regex pattern>
Result: <transformed text>
Matches: <number of matches found>
```

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `execute_skill_script` with pattern, replacement, and text arguments.
3. Present the transformation result to the user.
