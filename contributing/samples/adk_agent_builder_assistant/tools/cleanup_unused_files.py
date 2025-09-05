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

"""Cleanup unused files tool for Agent Builder Assistant."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


async def cleanup_unused_files(
    root_directory: str,
    used_files: List[str],
    file_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Identify and optionally delete unused files in project directories.

  This tool helps clean up unused tool files when agent configurations change.
  It identifies files that match patterns but aren't referenced in used_files
  list.

  Args:
    root_directory: Root directory to scan for unused files
    used_files: List of file paths currently in use (should not be deleted)
    file_patterns: List of glob patterns to match files (default: ["*.py"])
    exclude_patterns: List of patterns to exclude (default: ["__init__.py"])

  Returns:
    Dict containing cleanup results:
      - success: bool indicating if scan succeeded
      - root_directory: absolute path to scanned directory
      - unused_files: list of unused files found
      - deleted_files: list of files actually deleted
      - backup_files: list of backup files created
      - errors: list of error messages
      - total_freed_space: total bytes freed by deletions
  """
  try:
    root_path = Path(root_directory).resolve()
    used_files_set = {Path(f).resolve() for f in used_files}

    # Set defaults
    if file_patterns is None:
      file_patterns = ["*.py"]
    if exclude_patterns is None:
      exclude_patterns = ["__init__.py", "*_test.py", "test_*.py"]

    result = {
        "success": False,
        "root_directory": str(root_path),
        "unused_files": [],
        "deleted_files": [],
        "backup_files": [],
        "errors": [],
        "total_freed_space": 0,
    }

    if not root_path.exists():
      result["errors"].append(f"Root directory does not exist: {root_path}")
      return result

    # Find all files matching patterns
    all_files = []
    for pattern in file_patterns:
      all_files.extend(root_path.rglob(pattern))

    # Filter out excluded patterns
    for exclude_pattern in exclude_patterns:
      all_files = [f for f in all_files if not f.match(exclude_pattern)]

    # Identify unused files
    unused_files = []
    for file_path in all_files:
      if file_path not in used_files_set:
        unused_files.append(file_path)

    result["unused_files"] = [str(f) for f in unused_files]

    # Note: This function only identifies unused files
    # Actual deletion should be done with explicit user confirmation using delete_files()
    result["success"] = True

    return result

  except Exception as e:
    return {
        "success": False,
        "root_directory": root_directory,
        "unused_files": [],
        "deleted_files": [],
        "backup_files": [],
        "errors": [f"Cleanup scan failed: {str(e)}"],
        "total_freed_space": 0,
    }
