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

from __future__ import annotations

import abc
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
from ..agents.readonly_context import ReadonlyContext
from ..models.llm_request import LlmRequest


class BasePlanner(ABC):
  """Abstract base class for all planners.

  The planner allows the agent to generate plans for the queries to guide its
  action.
  """

  @abc.abstractmethod
  def build_planning_instruction(
      self,
      readonly_context: ReadonlyContext,
      llm_request: LlmRequest,
  ) -> Optional[str]:
    """Builds the system instruction to be appended to the LLM request for planning.

    Args:
        readonly_context: The readonly context of the invocation.
        llm_request: The LLM request. Readonly.

    Returns:
        The planning system instruction, or None if no instruction is needed.
    """
    pass

  @abc.abstractmethod
  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: List[types.Part],
  ) -> Optional[List[types.Part]]:
    """Processes the LLM response for planning.

    Args:
        callback_context: The callback context of the invocation.
        response_parts: The LLM response parts. Readonly.

    Returns:
        The processed response parts, or None if no processing is needed.
    """
    pass

  def to_content_blocks(
      self,
      response_parts: List[types.Part],
  ) -> List[Dict[str, Any]]:
    """Converts planner response parts to standardized content blocks.

    Produces a representation compatible with the standard content block format
    (similar to LangChain 1.0), where thought/reasoning parts are emitted as
    ``{'type': 'reasoning', 'reasoning': '...'}`` blocks and regular text parts
    as ``{'type': 'text', 'text': '...'}`` blocks.  Parts that are neither text
    nor thought are skipped.

    Args:
        response_parts: The planner response parts to convert.

    Returns:
        A list of standardized content block dicts.
    """
    blocks: List[Dict[str, Any]] = []
    for part in response_parts:
      if part.text is None:
        continue
      if part.thought:
        blocks.append({'type': 'reasoning', 'reasoning': part.text})
      else:
        blocks.append({'type': 'text', 'text': part.text})
    return blocks
