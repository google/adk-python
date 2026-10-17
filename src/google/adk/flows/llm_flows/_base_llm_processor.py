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

"""Defines the processor interface used for BaseLlmFlow."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from ...agents.invocation_context import InvocationContext
from ...events.event import Event

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse


class BaseLlmRequestProcessor(ABC):
  """Base class for LLM request processor."""

  name: str = ""
  """A stable identifier for this processor within a flow's request pipeline.

  Declaring a name lets callers locate this processor by name (across
  `request_processors` and `tool_request_processors`) instead of by list index
  or private class reference. Built-in processors set this at the class level;
  a processor built by a factory can also set it per instance.

  Defaults to `""`, which means the processor is anonymous and is ignored by
  name-based lookups.
  """

  @abstractmethod
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    """Runs the processor."""
    raise NotImplementedError("Not implemented.")
    yield  # AsyncGenerator requires a yield in function body.


class BaseLlmResponseProcessor(ABC):
  """Base class for LLM response processor."""

  name: str = ""
  """A stable identifier for this processor within `response_processors`.

  See `BaseLlmRequestProcessor.name`. Request and response pipelines are
  separate, so a response processor may share a name with a request processor
  (for example, `code_execution` and `nl_planning`).
  """

  @abstractmethod
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    """Processes the LLM response."""
    raise NotImplementedError("Not implemented.")
    yield  # AsyncGenerator requires a yield in function body.
