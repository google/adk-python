# Copyright 2025 Google LLC
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

"""ADK source code search tool for Agent Builder Assistant."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..utils import find_adk_source_folder


async def search_adk_source(
    search_pattern: str,
    file_patterns: Optional[List[str]] = None,
    max_results: int = 20,
    context_lines: int = 3,
    case_sensitive: bool = False,
) -> Dict[str, Any]:
  """Search ADK source code using regex patterns.

  This tool provides a regex-based alternative to vector-based retrieval for
  finding
  specific code patterns, class definitions, function signatures, and
  implementations
  in the ADK source code.

  Args:
    search_pattern: Regex pattern to search for (e.g., "class FunctionTool",
      "def __init__")
    file_patterns: List of glob patterns for files to search (default: ["*.py"])
    max_results: Maximum number of results to return (default: 20)
    context_lines: Number of context lines to include around matches (default:
      3)
    case_sensitive: Whether search should be case-sensitive (default: False)

  Returns:
    Dict containing search results:
      - success: bool indicating if search succeeded
      - pattern: the regex pattern used
      - total_matches: total number of matches found
      - files_searched: number of files searched
      - results: list of match results:
        - file_path: path to file containing match
        - line_number: line number of match
        - match_text: the matched text
        - context_before: lines before the match
        - context_after: lines after the match
        - full_match: complete context including before/match/after
      - errors: list of error messages
  """
  try:
    # Find ADK source directory dynamically
    adk_source_path = find_adk_source_folder()
    if not adk_source_path:
      return {
          "success": False,
          "pattern": search_pattern,
          "total_matches": 0,
          "files_searched": 0,
          "results": [],
          "errors": [
              "ADK source directory not found. Make sure you're running from"
              " within the ADK project."
          ],
      }

    adk_src_dir = Path(adk_source_path)

    result = {
        "success": False,
        "pattern": search_pattern,
        "total_matches": 0,
        "files_searched": 0,
        "results": [],
        "errors": [],
    }

    if not adk_src_dir.exists():
      result["errors"].append(f"ADK source directory not found: {adk_src_dir}")
      return result

    # Set default file patterns
    if file_patterns is None:
      file_patterns = ["*.py"]

    # Compile regex pattern
    try:
      flags = 0 if case_sensitive else re.IGNORECASE
      regex = re.compile(search_pattern, flags)
    except re.error as e:
      result["errors"].append(f"Invalid regex pattern: {str(e)}")
      return result

    # Find all Python files to search
    files_to_search = []
    for pattern in file_patterns:
      files_to_search.extend(adk_src_dir.rglob(pattern))

    result["files_searched"] = len(files_to_search)

    # Search through files
    for file_path in files_to_search:
      if result["total_matches"] >= max_results:
        break

      try:
        with open(file_path, "r", encoding="utf-8") as f:
          lines = f.readlines()

        for i, line in enumerate(lines):
          if result["total_matches"] >= max_results:
            break

          match = regex.search(line.rstrip())
          if match:
            # Get context lines
            start_line = max(0, i - context_lines)
            end_line = min(len(lines), i + context_lines + 1)

            context_before = [lines[j].rstrip() for j in range(start_line, i)]
            context_after = [lines[j].rstrip() for j in range(i + 1, end_line)]

            match_result = {
                "file_path": str(file_path.relative_to(adk_src_dir)),
                "line_number": i + 1,
                "match_text": line.rstrip(),
                "context_before": context_before,
                "context_after": context_after,
                "full_match": "\n".join(
                    context_before + [f">>> {line.rstrip()}"] + context_after
                ),
            }

            result["results"].append(match_result)
            result["total_matches"] += 1

      except Exception as e:
        result["errors"].append(f"Error searching {file_path}: {str(e)}")
        continue

    result["success"] = True
    return result

  except Exception as e:
    return {
        "success": False,
        "pattern": search_pattern,
        "total_matches": 0,
        "files_searched": 0,
        "results": [],
        "errors": [f"Search failed: {str(e)}"],
    }
