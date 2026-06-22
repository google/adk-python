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

"""Standardized content blocks for planner output.

Planners such as
:class:`~google.adk.planners.plan_re_act_planner.PlanReActPlanner` and
:class:`~google.adk.planners.built_in_planner.BuiltInPlanner` produce their
output as a list of ``google.genai.types.Part`` objects. Reasoning is signalled
on those parts with ``thought=True`` and, for ``PlanReActPlanner``, the text is
additionally annotated with inline ``/*PLANNING*/`` style tags that callers have
to parse themselves.

This module provides a small, additive helper that converts that ``Part`` list
into a provider-agnostic, structured representation modelled after LangChain
v1's standard content blocks
(https://docs.langchain.com/oss/python/langchain/messages#standard-content-blocks),
so that consumers can branch on a typed ``type`` discriminator instead of
re-parsing raw text.

The helper is read-only: it never mutates the parts it is given and it does not
change anything about how planners build instructions or post-process responses.
Existing consumers that rely on the ``Part`` output are unaffected.
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from google.genai import types
from typing_extensions import TypedDict


# Inline tags emitted by ``PlanReActPlanner``. Kept in sync with
# ``plan_re_act_planner`` but duplicated here to avoid a circular import and to
# keep this module self-contained.
_PLANNING_TAG = '/*PLANNING*/'
_REPLANNING_TAG = '/*REPLANNING*/'
_REASONING_TAG = '/*REASONING*/'
_ACTION_TAG = '/*ACTION*/'
_FINAL_ANSWER_TAG = '/*FINAL_ANSWER*/'

# Maps a leading PlanReActPlanner tag to the fine-grained reasoning kind exposed
# on reasoning content blocks.
_TAG_TO_REASONING_KIND: Dict[str, str] = {
    _PLANNING_TAG: 'planning',
    _REPLANNING_TAG: 'replanning',
    _REASONING_TAG: 'reasoning',
    _ACTION_TAG: 'action',
}

ReasoningKind = Literal['planning', 'replanning', 'reasoning', 'action']
"""Fine-grained category of a reasoning block.

For ``PlanReActPlanner`` this is derived from the inline tag that prefixes the
text (``/*PLANNING*/`` -> ``planning`` and so on). For ``BuiltInPlanner`` and
for thought parts without a recognized tag it is ``None``.
"""


class ReasoningContentBlock(TypedDict, total=False):
  """A reasoning (a.k.a. thought) content block.

  Modelled after LangChain v1's ``reasoning`` content block. ``type`` is always
  ``"reasoning"`` and ``reasoning`` carries the reasoning text.
  """

  type: Literal['reasoning']
  reasoning: str
  # ADK-specific, optional: which PlanReActPlanner phase this reasoning belongs
  # to, when it can be determined. ``None`` for built-in thinking.
  reasoning_kind: Optional[ReasoningKind]


class TextContentBlock(TypedDict, total=False):
  """A plain text content block (e.g. a planner's final answer).

  Modelled after LangChain v1's ``text`` content block.
  """

  type: Literal['text']
  text: str


class ToolCallContentBlock(TypedDict, total=False):
  """A tool/function call content block.

  Modelled after LangChain v1's ``tool_call`` content block. Carries the
  function call requested by the planner.
  """

  type: Literal['tool_call']
  name: str
  args: Dict[str, Any]
  # Provider-assigned call id, when present on the part.
  id: Optional[str]


# A standardized planner content block. New block ``type``s may be added in the
# future, so consumers should treat unknown types defensively.
ContentBlock = Dict[str, Any]


def _strip_leading_tag(text: str) -> tuple[str, Optional[str]]:
  """Splits a recognized leading PlanReActPlanner tag off ``text``.

  Args:
    text: The (already thought-marked) reasoning text.

  Returns:
    A ``(body, reasoning_kind)`` tuple. ``body`` is ``text`` with a single
    recognized leading tag removed and surrounding whitespace stripped;
    ``reasoning_kind`` is the matching kind, or ``None`` when no recognized tag
    prefixes the text.
  """
  stripped = text.lstrip()
  for tag, kind in _TAG_TO_REASONING_KIND.items():
    if stripped.startswith(tag):
      return stripped[len(tag):].strip(), kind
  return text, None


def part_to_content_block(part: types.Part) -> Optional[ContentBlock]:
  """Converts a single ``Part`` into a standardized content block.

  Args:
    part: The planner-produced part. Not mutated.

  Returns:
    A standardized content block, or ``None`` if the part carries no content
    that maps to a block (e.g. an empty part, or a redacted thought that only
    carries a signature).
  """
  # Function/tool calls take precedence over any incidental text on the part.
  if part.function_call and part.function_call.name:
    fc = part.function_call
    block: ToolCallContentBlock = {
        'type': 'tool_call',
        'name': fc.name,
        'args': dict(fc.args) if fc.args else {},
        'id': fc.id,
    }
    return block

  # Only text parts produce reasoning/text blocks. A thought part with no text
  # (e.g. an Anthropic redacted-thinking part that only carries a signature)
  # has no displayable content, so it is skipped.
  if not part.text:
    return None

  if part.thought:
    body, kind = _strip_leading_tag(part.text)
    # PlanReActPlanner splits a reasoning/final-answer part on the *last*
    # ``/*FINAL_ANSWER*/`` and keeps that separator on the trailing edge of the
    # reasoning part. It is a pure marker the planner already acted on, so drop
    # it from the standardized reasoning text.
    if body.endswith(_FINAL_ANSWER_TAG):
      body = body[: -len(_FINAL_ANSWER_TAG)].rstrip()
    reasoning_block: ReasoningContentBlock = {
        'type': 'reasoning',
        'reasoning': body,
        'reasoning_kind': kind,
    }
    return reasoning_block

  # A non-thought text part is final/answer text. PlanReActPlanner leaves the
  # ``/*FINAL_ANSWER*/`` tag out of this part already, but strip a stray leading
  # one defensively so the standardized block is clean.
  text = part.text
  stripped = text.lstrip()
  if stripped.startswith(_FINAL_ANSWER_TAG):
    text = stripped[len(_FINAL_ANSWER_TAG):].strip()
  text_block: TextContentBlock = {'type': 'text', 'text': text}
  return text_block


def parts_to_content_blocks(
    parts: Optional[List[types.Part]],
) -> List[ContentBlock]:
  """Converts planner-produced parts into standardized content blocks.

  This is the inverse-facing counterpart to a planner's
  ``process_planning_response``: given the parts a planner produced (where
  reasoning is signalled by ``thought=True`` and, for ``PlanReActPlanner``, by
  inline tags), it returns a provider-agnostic list of typed content blocks.

  The conversion is read-only and never mutates ``parts``. Parts that carry no
  mappable content are skipped, so the returned list may be shorter than
  ``parts``.

  Args:
    parts: The planner-produced parts, or ``None``.

  Returns:
    A list of standardized content blocks. Empty when ``parts`` is falsy or
    contains nothing mappable.
  """
  if not parts:
    return []
  blocks: List[ContentBlock] = []
  for part in parts:
    block = part_to_content_block(part)
    if block is not None:
      blocks.append(block)
  return blocks
