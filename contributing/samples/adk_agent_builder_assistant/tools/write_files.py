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

"""File writing tool for Agent Builder Assistant."""

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any
from typing import Dict

from google.adk.tools.tool_context import ToolContext

from ..utils.resolve_root_directory import resolve_file_path


async def write_files(
    files: Dict[str, str],
    tool_context: ToolContext,
    create_backup: bool = False,
    create_directories: bool = True,
) -> Dict[str, Any]:
  """Write content to multiple files with optional backup creation.

  This tool writes content to multiple files. It's designed for creating
  Python tools, callbacks, configuration files, and other code files.

  Args:
    files: Dict mapping file_path to content to write
    create_backup: Whether to create backups of existing files (default: False)
    create_directories: Whether to create parent directories (default: True)

  Returns:
    Dict containing write operation results:
      - success: bool indicating if all writes succeeded
      - files: dict mapping file_path to file info:
        - file_size: size of written file in bytes
        - existed_before: bool indicating if file existed before write
        - backup_created: bool indicating if backup was created
        - backup_path: path to backup file if created
        - error: error message if write failed for this file
      - successful_writes: number of files written successfully
      - total_files: total number of files requested
      - errors: list of general error messages
  """
  try:
    # Get session state for path resolution
    session_state = tool_context._invocation_context.session.state

    result = {
        "success": True,
        "files": {},
        "successful_writes": 0,
        "total_files": len(files),
        "errors": [],
    }

    for file_path, content in files.items():
      # Resolve file path using session state
      resolved_path = resolve_file_path(file_path, session_state)
      file_path_obj = resolved_path.resolve()
      file_info = {
          "file_size": 0,
          "existed_before": False,
          "backup_created": False,
          "backup_path": None,
          "error": None,
      }

      try:
        # Check if file already exists
        file_info["existed_before"] = file_path_obj.exists()

        # Create parent directories if needed
        if create_directories:
          file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if requested and file exists
        if create_backup and file_info["existed_before"]:
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

        # Write content to file
        with open(file_path_obj, "w", encoding="utf-8") as f:
          f.write(content)

        # Verify write and get file size
        if file_path_obj.exists():
          file_info["file_size"] = file_path_obj.stat().st_size
          result["successful_writes"] += 1
        else:
          file_info["error"] = "File was not created successfully"
          result["success"] = False

      except Exception as e:
        file_info["error"] = f"Write failed: {str(e)}"
        result["success"] = False

      result["files"][str(file_path_obj)] = file_info

    return result

  except Exception as e:
    return {
        "success": False,
        "files": {},
        "successful_writes": 0,
        "total_files": len(files) if files else 0,
        "errors": [f"Write operation failed: {str(e)}"],
    }
