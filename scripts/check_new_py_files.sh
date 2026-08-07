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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ADK_REAL_ROOT=$(dirname "$(realpath "$REPO_ROOT/src/google/adk/__init__.py")")
EXCLUDE_TESTS="$ADK_REAL_ROOT/tests"
EXCLUDE_WORKSPACE="$ADK_REAL_ROOT/open_source_workspace"
EXCLUDE_CONTRIBUTING="$ADK_REAL_ROOT/contributing"

DOCS_GUIDES_DIR="$REPO_ROOT/docs/guides"

# File and directory glob patterns exempt from the unit guide requirement.
EXEMPT_GUIDE_PATTERNS=(
    "__init__.py"
    "cli/*" "*/cli/*"
    "utils/*" "*/utils/*"
    "*_utils.py"
    "*_helper.py" "*_helpers.py"
    "*_types.py"
    "*_errors.py" "*_exceptions.py"
    "*_constants.py"
)

is_exempt_from_unit_guide() {
  local rel_path="$1"
  local filename="$2"
  local pattern
  for pattern in "${EXEMPT_GUIDE_PATTERNS[@]}"; do
    if [[ "$rel_path" == $pattern ]] || [[ "$filename" == $pattern ]]; then
      return 0
    fi
  done
  return 1
}

exit_code=0

get_added_files() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    staged=$(git diff --cached --name-only --diff-filter=A 2>/dev/null)
    if [[ -n "$staged" ]]; then
      echo "$staged"
    else
      git diff HEAD~1..HEAD --name-only --diff-filter=A 2>/dev/null
    fi
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

get_commit_message() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    msg=$(git log -1 --pretty=%B 2>/dev/null || true)
    git_dir=$(git rev-parse --git-dir 2>/dev/null || echo "")
    if [[ -n "$git_dir" && -f "$git_dir/COMMIT_EDITMSG" ]]; then
      msg="$msg $(cat "$git_dir/COMMIT_EDITMSG" 2>/dev/null || true)"
    fi
    echo "$msg"
  elif jj root >/dev/null 2>&1; then
    jj log -r @ --no-graph -T description 2>/dev/null
  elif hg root >/dev/null 2>&1; then
    hg log -r . --template '{desc}' 2>/dev/null
  elif g4 info >/dev/null 2>&1; then
    g4 change -o 2>/dev/null || g4 describe 2>/dev/null
  elif p4 info >/dev/null 2>&1; then
    p4 change -o 2>/dev/null
  fi
}

commit_msg=$(get_commit_message)
has_no_unit_guide_tag=false
if [[ -n "${NO_UNIT_GUIDE:-}" ]] || [[ -n "${SKIP_UNIT_GUIDE:-}" ]] || echo "$commit_msg" | grep -q -i -E "NO_UNIT_GUIDE|SKIP_UNIT_GUIDE"; then
    has_no_unit_guide_tag=true
fi

while read -r file; do
    # Check if file is not empty (happens if no new files)
    if [[ -n "$file" ]]; then
        # Match only files in the package source (resolving symlinks to support both
        # open-source and internal layouts) and exclude tests/workspace to avoid
        # false positives.
        abs_file=$(realpath "$file" 2>/dev/null || echo "")
        if [[ -n "$abs_file" ]] && \
           [[ "$abs_file" == "$ADK_REAL_ROOT"/* ]] && \
           [[ "$abs_file" != "$EXCLUDE_TESTS"/* ]] && \
           [[ "$abs_file" != "$EXCLUDE_WORKSPACE"/* ]] && \
           [[ "$abs_file" != "$EXCLUDE_CONTRIBUTING"/* ]] && \
           [[ "$abs_file" == *.py ]]; then
            filename=$(basename "$abs_file")

            # Check 1: Enforce private '_' prefix rule
            if [[ ! "$filename" == _* ]]; then
                echo "Error: New Python file '$file' must have a '_' prefix."
                echo "All new Python files in src/google/adk/ must be private by default."
                echo "To expose a public interface, use __init__.py and list public symbols in __all__."
                echo "See .agents/skills/adk-style/references/visibility.md for details."
                exit_code=1
            fi

            # Check 2: Enforce unit guide rule
            rel_path="${abs_file#$ADK_REAL_ROOT/}"
            rel_dir=$(dirname "$rel_path")

            if ! is_exempt_from_unit_guide "$rel_path" "$filename" && [[ "$has_no_unit_guide_tag" == false ]]; then
                name_no_ext="${filename%.py}"
                name_no_prefix="${name_no_ext#_}"

                guide_found=false
                # Check candidate paths in docs/guides
                for cand_name in "$name_no_prefix" "$name_no_ext"; do
                    if [[ "$rel_dir" != "." ]]; then
                        if [[ -f "$DOCS_GUIDES_DIR/$rel_dir/$cand_name/index.md" ]] || \
                           [[ -f "$DOCS_GUIDES_DIR/$rel_dir/$cand_name.md" ]]; then
                            guide_found=true
                            break
                        fi
                    else
                        if [[ -f "$DOCS_GUIDES_DIR/$cand_name/index.md" ]] || \
                           [[ -f "$DOCS_GUIDES_DIR/$cand_name.md" ]]; then
                            guide_found=true
                            break
                        fi
                    fi
                done

                if [[ "$guide_found" == false ]]; then
                    echo "Error: New Python file '$file' requires a unit guide in docs/guides/."
                    if [[ "$rel_dir" != "." ]]; then
                        echo "Expected guide at 'docs/guides/$rel_dir/$name_no_prefix/index.md' or 'docs/guides/$rel_dir/$name_no_prefix.md'."
                    else
                        echo "Expected guide at 'docs/guides/$name_no_prefix/index.md' or 'docs/guides/$name_no_prefix.md'."
                    fi
                    echo "If a unit guide is not required for this file, add a tag in your commit message/CL description explaining why (e.g. 'NO_UNIT_GUIDE=<reason>')."
                    echo "See .agents/skills/adk-unit-guide/SKILL.md for details on creating unit guides."
                    exit_code=1
                fi
            fi
        fi
    fi
done < <(get_added_files)

exit $exit_code

