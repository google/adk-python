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

"""Working directory helper tool to resolve path context issues."""

import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional


async def resolve_root_directory(
    root_directory: str, working_directory: Optional[str] = None
) -> Dict[str, Any]:
  """Resolve the root directory from user-provided path for agent building.

  This tool determines where to create or update agent configurations by
  resolving the user-provided path. It handles both absolute and relative paths,
  using the current working directory when needed for relative path resolution.

  Args:
    root_directory: Path provided by user (can be relative or absolute)
      indicating where to build agents
    working_directory: Optional explicit working directory to use as base for
      relative path resolution (defaults to os.getcwd())

  Returns:
    Dict containing path resolution results:
      Always included:
        - success: bool indicating if resolution succeeded
        - original_path: the provided root directory path
        - resolved_path: absolute path to the resolved location
        - resolution_method: explanation of how path was resolved
        - path_exists: bool indicating if resolved path exists

      Conditionally included:
        - alternative_paths: list of other possible path interpretations
        - warnings: list of potential issues or ambiguities
        - working_directory_used: the working directory used for resolution

  Examples:
    Resolve relative path:
      result = await resolve_root_directory("./my_project",
      "/home/user/projects")

    Resolve with auto-detection:
      result = await resolve_root_directory("my_agent.yaml")
      # Will use current working directory for relative paths
  """
  try:
    current_cwd = os.getcwd()
    root_path_obj = Path(root_directory)

    # If user provided an absolute path, use it directly
    if root_path_obj.is_absolute():
      resolved_path = root_path_obj
    else:
      # For relative paths, prefer user-provided working directory
      if working_directory:
        resolved_path = Path(working_directory) / root_directory
      else:
        # Fallback to actual current working directory
        resolved_path = Path(current_cwd) / root_directory

    return {
        "success": True,
        "original_path": root_directory,
        "resolved_path": str(resolved_path.resolve()),
        "exists": resolved_path.exists(),
        "is_absolute": root_path_obj.is_absolute(),
        "current_cwd": current_cwd,
        "working_directory_used": working_directory,
        "recommendation": (
            f"Use resolved path: {resolved_path.resolve()}"
            if resolved_path.exists()
            else (
                "Path does not exist. Create parent directories first:"
                f" {resolved_path.parent}"
            )
        ),
    }

  except Exception as e:
    return {
        "success": False,
        "error": f"Failed to resolve path: {str(e)}",
        "original_path": root_directory,
    }
