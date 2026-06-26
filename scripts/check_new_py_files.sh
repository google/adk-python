#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
