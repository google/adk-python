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
from typing import List
from typing import Optional


def resolve_file_path(
    file_path: str,
    session_state: Optional[Dict[str, Any]] = None,
    working_directory: Optional[str] = None,
) -> Path:
  """Resolve a file path using root directory from session state.

  This is a helper function that other tools can use to resolve file paths
  without needing to be async or return detailed resolution information.

  Args:
    file_path: File path (relative or absolute)
    session_state: Session state dict that may contain root_directory
    working_directory: Working directory to use as base (defaults to cwd)

  Returns:
    Resolved absolute Path object
  """
  file_path_obj = Path(file_path)

  # If already absolute, use as-is
  if file_path_obj.is_absolute():
    return file_path_obj

  # Get root directory from session state, default to "./"
  root_directory = "./"
  if session_state and "root_directory" in session_state:
    root_directory = session_state["root_directory"]

  # Use the same resolution logic as the main function
  root_path_obj = Path(root_directory)

  if root_path_obj.is_absolute():
    resolved_root = root_path_obj
  else:
    if working_directory:
      resolved_root = Path(working_directory) / root_directory
    else:
      resolved_root = Path(os.getcwd()) / root_directory

  # Resolve file path relative to root directory
  return resolved_root / file_path


def resolve_file_paths(
    file_paths: List[str],
    session_state: Optional[Dict[str, Any]] = None,
    working_directory: Optional[str] = None,
) -> List[Path]:
  """Resolve multiple file paths using root directory from session state.

  Args:
    file_paths: List of file paths (relative or absolute)
    session_state: Session state dict that may contain root_directory
    working_directory: Working directory to use as base (defaults to cwd)

  Returns:
    List of resolved absolute Path objects
  """
  return [
      resolve_file_path(path, session_state, working_directory)
      for path in file_paths
  ]
