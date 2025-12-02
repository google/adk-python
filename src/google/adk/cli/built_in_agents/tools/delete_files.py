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

"""File deletion tool for Agent Builder Assistant."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any
from typing import Dict
from typing import List

from google.adk.tools.tool_context import ToolContext

from ..utils.resolve_root_directory import resolve_file_paths


async def delete_files(
    file_paths: List[str],
    tool_context: ToolContext,
    create_backup: bool = False,
    confirm_deletion: bool = True,
) -> Dict[str, Any]:
  """Delete multiple files with optional backup creation.

  This tool safely deletes multiple files with validation and optional backup
  creation.
  It's designed for cleaning up unused tool files when agent configurations
  change.

  Args:
    file_paths: List of absolute or relative paths to files to delete
    create_backup: Whether to create a backup before deletion (default: False)
    confirm_deletion: Whether deletion was confirmed by user (default: True for
      safety)

  Returns:
    Dict containing deletion operation results:
      - success: bool indicating if all deletions succeeded
      - files: dict mapping file_path to file deletion info:
        - existed: bool indicating if file existed before deletion
        - backup_created: bool indicating if backup was created
        - backup_path: path to backup file if created
        - error: error message if deletion failed for this file
        - file_size: size of deleted file in bytes (if existed)
      - successful_deletions: number of files deleted successfully
      - total_files: total number of files requested
      - errors: list of general error messages
  """
  try:
    # Resolve file paths using session state
    session_state = tool_context._invocation_context.session.state
    resolved_paths = resolve_file_paths(file_paths, session_state)

    result = {
        "success": True,
        "files": {},
        "successful_deletions": 0,
        "total_files": len(file_paths),
        "errors": [],
    }

    # Safety check - only delete if user confirmed
    if not confirm_deletion:
      result["success"] = False
      result["errors"].append("Deletion not confirmed by user")
      return result

    for resolved_path in resolved_paths:
      file_path_obj = resolved_path.resolve()
      file_info = {
          "existed": False,
          "backup_created": False,
          "backup_path": None,
          "error": None,
          "file_size": 0,
      }

      try:
        # Check if file exists
        if not file_path_obj.exists():
          file_info["error"] = f"File does not exist: {file_path_obj}"
          result["files"][str(file_path_obj)] = file_info
          result["successful_deletions"] += 1  # Still count as success
          continue

        file_info["existed"] = True
        file_info["file_size"] = file_path_obj.stat().st_size

        # Create backup if requested
        if create_backup:
          timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
          backup_path = file_path_obj.with_suffix(
              f".backup_{timestamp}{file_path_obj.suffix}"
          )
          try:
            shutil.copy2(file_path_obj, backup_path)
            file_info["backup_created"] = True
            file_info["backup_path"] = str(backup_path)
          except Exception as e:
            file_info["error"] = f"Failed to create backup: {str(e)}"
            result["success"] = False
            result["files"][str(file_path_obj)] = file_info
            continue

        # Delete the file
        file_path_obj.unlink()
        result["successful_deletions"] += 1

      except Exception as e:
        file_info["error"] = f"Deletion failed: {str(e)}"
        result["success"] = False

      result["files"][str(file_path_obj)] = file_info

    return result

  except Exception as e:
    return {
        "success": False,
        "files": {},
        "successful_deletions": 0,
        "total_files": len(file_paths) if file_paths else 0,
        "errors": [f"Delete operation failed: {str(e)}"],
    }
