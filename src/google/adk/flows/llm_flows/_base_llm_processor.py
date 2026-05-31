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

  @abstractmethod
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    """Runs the processor."""
    raise NotImplementedError("Not implemented.")
    yield  # AsyncGenerator requires a yield in function body.


class BaseLlmResponseProcessor(ABC):
  """Base class for LLM response processor."""

  @abstractmethod
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    """Processes the LLM response."""
    raise NotImplementedError("Not implemented.")
    yield  # AsyncGenerator requires a yield in function body.
