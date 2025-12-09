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


"""
Utility functions for Google Cloud resource management.

Solves common issues with resource ID extraction for ADK users.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_agent_engine_id(agent_engine_resource_name: str) -> str:
  """
  Extract Agent Engine ID from Vertex AI Agent Engine resource name.

  Solves issue #2940: Users cannot find agent ID for Memory Bank Service.
  The api_resource.name returns full path, but Memory Bank needs only the ID.

  Args:
      agent_engine_resource_name (str): Full Agent Engine resource path
          Example: "projects/my-project/locations/us-central1/agentEngines/abc123"

  Returns:
      str: Agent Engine ID (e.g., "abc123")

  Raises:
      ValueError: If resource name format is invalid or empty

  Example:
      >>> resource_name = "projects/test/locations/us-central1/agentEngines/abc123def"
      >>> extract_agent_engine_id(resource_name)
      'abc123def'
  """
  if not agent_engine_resource_name:
    raise ValueError("Agent Engine resource name cannot be empty")

  if not isinstance(agent_engine_resource_name, str):
    raise ValueError(
        f"Resource name must be string, got {type(agent_engine_resource_name)}"
    )

  # Extract the last segment of the path
  agent_id = agent_engine_resource_name.split("/")[-1]

  if not agent_id:
    raise ValueError(
        "Could not extract agent ID from resource name:"
        f" {agent_engine_resource_name}"
    )

  # Validate format - should be alphanumeric with hyphens
  if not re.match(r"^[a-zA-Z0-9\-]+$", agent_id):
    logger.warning(f"Agent ID '{agent_id}' contains non-standard characters")

  logger.info(f"Successfully extracted Agent ID: {agent_id}")
  return agent_id


def validate_agent_engine_resource_name(resource_name: str) -> bool:
  """
  Validate Agent Engine resource name format.

  Args:
      resource_name: Resource name to validate

  Returns:
      True if valid format, False otherwise
  """
  if not resource_name or not isinstance(resource_name, str):
    return False

  # Accept both agentEngines and reasoningEngines patterns
  pattern = r"^projects/[^/]+/locations/[^/]+/(agentEngines|reasoningEngines)/[a-zA-Z0-9\-]+$"
  return bool(re.match(pattern, resource_name))


def get_project_from_resource_name(resource_name: str) -> Optional[str]:
  """Extract project ID from resource name."""
  try:
    parts = resource_name.split("/")
    if len(parts) >= 2 and parts[0] == "projects":
      return parts[1]
  except Exception:
    pass
  return None


def get_location_from_resource_name(resource_name: str) -> Optional[str]:
  """Extract location from resource name."""
  try:
    parts = resource_name.split("/")
    if len(parts) >= 4 and parts[2] == "locations":
      return parts[3]
  except Exception:
    pass
  return None
