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

"""File reading tool for Agent Builder Assistant."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


async def read_files(file_paths: List[str]) -> Dict[str, Any]:
  """Read content from multiple files.

  This tool reads content from multiple files and returns their contents.
  It's designed for reading Python tools, configuration files, and other text
  files.

  Args:
    file_paths: List of absolute or relative paths to files to read

  Returns:
    Dict containing read operation results:
      - success: bool indicating if all reads succeeded
      - files: dict mapping file_path to file info:
        - content: file content as string
        - file_size: size of file in bytes
        - exists: bool indicating if file exists
        - error: error message if read failed for this file
      - successful_reads: number of files read successfully
      - total_files: total number of files requested
      - errors: list of general error messages
  """
  try:
    result = {
        "success": True,
        "files": {},
        "successful_reads": 0,
        "total_files": len(file_paths),
        "errors": [],
    }

    for file_path in file_paths:
      file_path_obj = Path(file_path).resolve()
      file_info = {
          "content": "",
          "file_size": 0,
          "exists": False,
          "error": None,
      }

      try:
        if not file_path_obj.exists():
          file_info["error"] = f"File does not exist: {file_path_obj}"
        else:
          file_info["exists"] = True
          file_info["file_size"] = file_path_obj.stat().st_size

          with open(file_path_obj, "r", encoding="utf-8") as f:
            file_info["content"] = f.read()

          result["successful_reads"] += 1
      except Exception as e:
        file_info["error"] = f"Failed to read {file_path}: {str(e)}"
        result["success"] = False

      result["files"][str(file_path_obj)] = file_info

    return result

  except Exception as e:
    return {
        "success": False,
        "files": {},
        "successful_reads": 0,
        "total_files": len(file_paths) if file_paths else 0,
        "errors": [f"Read operation failed: {str(e)}"],
    }
