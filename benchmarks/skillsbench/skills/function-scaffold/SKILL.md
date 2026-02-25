---
name: function-scaffold
description: Generate Python function scaffolds with type hints and docstrings from a specification.
---

# Function Scaffold Skill

Generate Python function stubs from a natural-language specification, including type hints, docstrings, and placeholder implementations.

## Available Scripts

### `scaffold.py`

Generates a Python function scaffold from a specification.

**Usage**: `run_skill_script(skill_name="function-scaffold", script_path="scripts/scaffold.py", args={"name": "calculate_bmi", "params": "weight:float,height:float", "returns": "float", "description": "Calculate Body Mass Index"})`

Arguments:
- `name`: Function name (snake_case)
- `params`: Comma-separated parameter list with types (e.g., `x:int,y:str`)
- `returns`: Return type annotation
- `description`: One-line description for the docstring

**Output format**: Complete Python function with type hints and docstring.

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `run_skill_script` with the function specification.
3. Present the generated scaffold to the user.
