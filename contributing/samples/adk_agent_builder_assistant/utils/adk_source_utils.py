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

"""Utilities for finding ADK source folder dynamically and loading schema."""

import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

# Set up logger for ADK source utils
logger = logging.getLogger("google_adk." + __name__)

# Global cache for ADK AgentConfig schema to avoid repeated file reads
_schema_cache: Optional[Dict[str, Any]] = None


def find_adk_source_folder(start_path: Optional[str] = None) -> Optional[str]:
  """Find the ADK source folder by searching up the directory tree.

  Searches for either 'src/google/adk' or 'google/adk' directories starting
  from the given path and moving up the directory tree until the root.

  Args:
    start_path: Directory to start search from. If None, uses current directory.

  Returns:
    Absolute path to the ADK source folder if found, None otherwise.

  Examples:
    Find ADK source from current directory:
      adk_path = find_adk_source_folder()

    Find ADK source from specific directory:
      adk_path = find_adk_source_folder("/path/to/project")
  """
  if start_path is None:
    start_path = os.path.dirname(__file__)

  current_path = Path(start_path).resolve()

  # Search patterns to look for
  search_patterns = ["src/google/adk", "google/adk"]

  logger.debug("Searching for ADK source from directory: %s", current_path)
  # Search up the directory tree until root
  while current_path != current_path.parent:  # Not at filesystem root
    for pattern in search_patterns:
      candidate_path = current_path / pattern
      if candidate_path.exists() and candidate_path.is_dir():
        # Verify it's actually an ADK source by checking for key files
        if _verify_adk_source_folder(candidate_path):
          return str(candidate_path)
    # Move to parent directory
    current_path = current_path.parent

  # Check root directory as well
  for pattern in search_patterns:
    candidate_path = current_path / pattern
    if candidate_path.exists() and candidate_path.is_dir():
      if _verify_adk_source_folder(candidate_path):
        logger.info("Found ADK source folder : %s", candidate_path)
        return str(candidate_path)
  return None


def _verify_adk_source_folder(path: Path) -> bool:
  """Verify that a path contains ADK source code.

  Args:
    path: Path to check

  Returns:
    True if path appears to contain ADK source code
  """
  # Check for key ADK source files/directories
  expected_items = ["agents/config_schemas/AgentConfig.json"]

  found_items = 0
  for item in expected_items:
    if (path / item).exists():
      found_items += 1

  return found_items == len(expected_items)


def get_adk_schema_path(start_path: Optional[str] = None) -> Optional[str]:
  """Find the path to the ADK AgentConfig schema file.

  Args:
    start_path: Directory to start search from. If None, uses current directory.

  Returns:
    Absolute path to AgentConfig.json schema file if found, None otherwise.
  """
  adk_source_path = find_adk_source_folder(start_path)
  if not adk_source_path:
    return None

  schema_path = Path(adk_source_path) / "agents/config_schemas/AgentConfig.json"
  if schema_path.exists() and schema_path.is_file():
    return str(schema_path)

  return None


def load_agent_config_schema(
    raw_format: bool = False, escape_braces: bool = False
) -> str | Dict[str, Any]:
  """Load the ADK AgentConfig.json schema with various formatting options.

  This function provides a centralized way to load the ADK AgentConfig schema
  and format it for different use cases across the Agent Builder Assistant.

  Args:
    raw_format: If True, return as JSON string. If False, return as parsed dict.
    escape_braces: If True, replace { and } with {{ and }} for template
      embedding. Only applies when raw_format=True.

  Returns:
    Either the ADK AgentConfig schema as a Dict (raw_format=False) or as a
    formatted string (raw_format=True), optionally with escaped braces for
    template use.

  Raises:
    FileNotFoundError: If ADK AgentConfig.json schema file is not found.

  Examples:
    # Get parsed ADK AgentConfig schema dict for validation
    schema_dict = load_agent_config_schema()

    # Get raw ADK AgentConfig schema JSON string for display
    schema_str = load_agent_config_schema(raw_format=True)

    # Get template-safe ADK AgentConfig schema JSON string for instruction
    # embedding
    schema_template = load_agent_config_schema(
        raw_format=True, escape_braces=True
    )
  """
  global _schema_cache

  # Load and cache schema if not already loaded
  if _schema_cache is None:
    schema_path_str = get_adk_schema_path()
    if not schema_path_str:
      raise FileNotFoundError(
          "AgentConfig.json schema not found. Make sure you're running from"
          " within the ADK project."
      )

    schema_path = Path(schema_path_str)
    if not schema_path.exists():
      raise FileNotFoundError(
          f"AgentConfig.json schema not found at {schema_path}"
      )

    with open(schema_path, "r", encoding="utf-8") as f:
      _schema_cache = json.load(f)

  # Return parsed dict format
  if not raw_format:
    return _schema_cache

  # Return as JSON string with optional brace escaping
  schema_str = json.dumps(_schema_cache, indent=2)

  if escape_braces:
    # Replace braces for template embedding (prevent variable interpolation)
    schema_str = schema_str.replace("{", "{{").replace("}", "}}")

  return schema_str


def clear_schema_cache() -> None:
  """Clear the cached schema data.

  This can be useful for testing or if the schema file has been updated
  and you need to reload it.
  """
  global _schema_cache
  _schema_cache = None
