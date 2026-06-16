# Formatting Style Guide

- 2-space indentation (never tabs).
- 80-character line limit.
- `pyink` formatter (Google-style).
- `isort` with Google profile for import sorting.
- Enforced automatically by pre-commit hooks (`isort`, `pyink`, `addlicense`, `mdformat`). Use the `adk-setup` skill to install and configure these tools.

## Running Formatter Manually

```bash
# Format only staged files (runs automatically on commit)
pre-commit run

# Format all changed files (staged + unstaged)
pre-commit run --files $(git diff --name-only HEAD)

# Format all files in the repo
pre-commit run --all-files
```
