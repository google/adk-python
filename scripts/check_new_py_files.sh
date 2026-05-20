#!/bin/bash

exit_code=0

# Get list of newly added files using diff-filter=A
# Using process substitution to avoid subshell and handle spaces in filenames
while read -r file; do
    # Check if file is not empty (happens if no new files)
    if [[ -n "$file" ]]; then
        if [[ "$file" == src/google/adk/*.py ]]; then
            filename=$(basename "$file")
            if [[ ! "$filename" == _* ]]; then
                echo "Error: New Python file '$file' must have a '_' prefix."
                echo "All new Python files in src/google/adk/ must be private by default."
                echo "To expose a public interface, use __init__.py and list public symbols in __all__."
                echo "See .agents/skills/adk-style/references/visibility.md for details."
                exit_code=1
            fi
        fi
    fi
done < <(git diff --cached --name-only --diff-filter=A)

exit $exit_code
