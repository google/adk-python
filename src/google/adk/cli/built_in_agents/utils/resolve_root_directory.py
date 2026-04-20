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

"""Working directory helper tool to resolve path context issues."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .path_normalizer import sanitize_generated_file_path


def _validate_root_directory(root_directory: str) -> None:
  """Validate that root_directory from session state is safe to use.

  Rejects values that could redirect file operations outside the project root.

  Args:
    root_directory: The root_directory value from session state.

  Raises:
    ValueError: If root_directory contains unsafe path components.
  """
  if not root_directory:
    return
  if Path(root_directory).is_absolute():
    raise ValueError(
        f'root_directory must be a relative path, got: {root_directory!r}'
    )
  if any(c in root_directory for c in ['\x00', '\\']):
    raise ValueError(
        f'root_directory contains invalid characters: {root_directory!r}'
    )
  if any(part == '..' for part in Path(root_directory).parts):
    raise ValueError(
        f"root_directory must not contain '..': {root_directory!r}"
    )


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

  Raises:
    ValueError: If the resolved path escapes the project root.
  """
  normalized_path = sanitize_generated_file_path(file_path)
  file_path_obj = Path(normalized_path)

  # Get root directory from session state, default to "./"
  root_directory = './'
  if session_state and 'root_directory' in session_state:
    root_directory = session_state['root_directory']
    _validate_root_directory(root_directory)

  # Compute the resolved root as an absolute path
  root_path_obj = Path(root_directory)
  if working_directory:
    resolved_root = (Path(working_directory) / root_path_obj).resolve()
  else:
    resolved_root = (Path(os.getcwd()) / root_path_obj).resolve()

  # Resolve the candidate path
  if file_path_obj.is_absolute():
    candidate = file_path_obj.resolve()
  else:
    candidate = (resolved_root / file_path_obj).resolve()

  # Enforce boundary: reject paths that escape the project root
  try:
    candidate.relative_to(resolved_root)
  except ValueError as e:
    raise ValueError(
        f'Path {file_path!r} resolves outside project root'
        f' {resolved_root!r}'
    ) from e

  return candidate


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
