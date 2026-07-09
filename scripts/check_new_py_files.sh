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

get_added_files() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --cached --name-only --diff-filter=A
  elif jj root >/dev/null 2>&1; then
    jj diff --summary 2>/dev/null | awk '/^A / {print $2}'
  elif hg root >/dev/null 2>&1; then
    hg status --added --no-status 2>/dev/null
  elif g4 info >/dev/null 2>&1; then
    g4 opened 2>/dev/null | awk '/ - add / {print $1}' | sed 's/#.*//'
  elif p4 info >/dev/null 2>&1; then
    p4 opened 2>/dev/null | awk '/ - add / {print $1}' | sed 's/#.*//'
  fi
}

while read -r file; do
    # Check if file is not empty (happens if no new files)
    if [[ -n "$file" ]]; then
        if [[ "$file" == */google/adk/*.py ]] || [[ "$file" == google/adk/*.py ]]; then
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
done < <(get_added_files)

exit $exit_code
