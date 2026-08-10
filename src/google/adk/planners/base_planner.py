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
      self, parts: List[types.Part]
  ) -> List[Dict[str, Any]]:
    """Converts response parts to standardized content blocks.

    Produces a list of dicts with a ``type`` key, following the standard
    content block convention used by LangChain and similar frameworks.
    Parts marked as thoughts are emitted as
    ``{'type': 'reasoning', 'reasoning': '<text>'}``; plain text parts are
    emitted as ``{'type': 'text', 'text': '<text>'}``;  function-call parts
    are skipped.

    This method works uniformly for both :class:`PlanReActPlanner` (where
    reasoning is identified by the ``thought`` flag) and
    :class:`BuiltInPlanner` (where the model sets ``thought=True`` natively).

    Args:
        parts: The response parts to convert, typically the return value of
          :meth:`process_planning_response`.

    Returns:
        A list of standardized content block dicts.
    """
    blocks: List[Dict[str, Any]] = []
    for part in parts:
      if part.thought and part.text:
        blocks.append({'type': 'reasoning', 'reasoning': part.text})
      elif part.text and not part.thought:
        blocks.append({'type': 'text', 'text': part.text})
    return blocks
