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

"""Data models for task-mode LlmAgent delegation.

Used by ``FinishTaskTool`` to validate and serialize task input/result
payloads.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
  """A request to delegate a task to a sub-agent."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  agent_name: str
  """The name of the target agent to delegate to."""

  input: dict[str, Any]
  """The validated input data for the task."""


class TaskResult(BaseModel):
  """The result returned by a task agent upon completion."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  output: Any
  """The validated output data from the task."""


def _as_task_request(value: Any) -> TaskRequest:
  """Convert a value to a TaskRequest instance.

  Handles both TaskRequest instances (same-invocation, stored directly)
  and plain dicts (after session deserialization via model_dump()).

  Args:
    value: A TaskRequest instance or a dict representation.

  Returns:
    A TaskRequest instance.
  """
  if isinstance(value, TaskRequest):
    return value
  if not isinstance(value, dict):
    logger.error(
        'Unexpected type for TaskRequest: %s. Expected TaskRequest or dict.',
        type(value).__name__,
    )
  return TaskRequest.model_validate(value)


class _DefaultTaskInput(BaseModel):
  """Default input schema when no custom input_schema is provided.

  Used by RequestTaskTool to generate the function declaration when the
  target agent does not define an explicit input_schema.
  """

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  goal: Optional[str] = None
  """The goal or objective for the task agent."""

  background: Optional[str] = None
  """Additional background context for the task agent."""


class _DefaultTaskOutput(BaseModel):
  """Default output schema when no custom output_schema is provided.

  Used by FinishTaskTool to generate the function declaration when the
  task agent does not define an explicit output_schema.
  """

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  result: str
  """A brief summary of what the agent accomplished."""
