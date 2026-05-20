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

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from google.adk.errors.input_validation_error import InputValidationError

from ._path_normalizer import _sanitize_generated_file_path as sanitize_generated_file_path
from ._path_normalizer import _to_posix_path as to_posix_path


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
    InputValidationError: If the path resolves outside the configured root
      directory.
  """
  normalized_path = sanitize_generated_file_path(file_path)
  if not normalized_path:
    raise InputValidationError("File path must not be empty.")

  resolved_root = _resolve_root_directory(session_state, working_directory)
  pure_path = to_posix_path(normalized_path)
  candidate_path = Path(pure_path)

  if candidate_path.is_absolute():
    resolved_path = candidate_path.resolve(strict=False)
  else:
    resolved_path = (resolved_root / candidate_path).resolve(strict=False)

  try:
    resolved_path.relative_to(resolved_root)
  except ValueError as exc:
    raise InputValidationError(
        f"File path {file_path!r} resolves outside root directory"
        f" {resolved_root}"
    ) from exc

  return resolved_path


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


def _resolve_root_directory(
    session_state: Optional[Dict[str, Any]] = None,
    working_directory: Optional[str] = None,
) -> Path:
  """Resolve the effective root directory for built-in agent file tools."""
  root_directory = "./"
  if session_state and "root_directory" in session_state:
    root_directory = str(session_state["root_directory"])

  normalized_root = sanitize_generated_file_path(root_directory) or "./"
  root_path_obj = Path(to_posix_path(normalized_root))

  if root_path_obj.is_absolute():
    return root_path_obj.resolve(strict=False)

  base_directory = Path(working_directory) if working_directory else Path.cwd()
  return (base_directory / root_path_obj).resolve(strict=False)
