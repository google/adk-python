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

import enum
import statistics
from typing import Any
from typing import Optional
from typing import Union

from google.genai import types as genai_types

from .app_details import AppDetails
from .common import EvalBaseModel
from .eval_case import get_all_tool_calls_with_responses
from .eval_case import IntermediateDataType
from .eval_case import Invocation
from .eval_case import InvocationEvents
from .eval_metrics import RubricScore
from .evaluator import EvalStatus


@enum.unique
class Label(enum.Enum):
  """Labels for auto rater response."""

  TRUE = "true"
  INVALID = "invalid"
  VALID = "valid"
  PARTIALLY_VALID = "partially_valid", "partially valid", "partially"
  ALMOST = "almost"
  FALSE = "false"
  NOT_FOUND = "label field not found"


def get_text_from_content(
    content: Optional[Union[genai_types.Content, Invocation]],
    *,
    include_intermediate_responses_in_final: bool = False,
) -> Optional[str]:
  """Extracts text from a `Content` or an `Invocation`.

  When `content` is a `Content`, returns the concatenated text of its parts.

  When `content` is an `Invocation`, returns the text of the invocation's final
  response. If `include_intermediate_responses_in_final` is True, text from
  intermediate invocation events (e.g. natural language emitted before tool
  calls) is concatenated with the final response text.
  """
  if isinstance(content, Invocation):
    if not include_intermediate_responses_in_final:
      # Flag off: revert to basic plain-Content behavior.
      return get_text_from_content(content.final_response)

    parts: list[str] = []
    if isinstance(content.intermediate_data, InvocationEvents):
      # Walk intermediate events in order; collect text parts.
      for event in content.intermediate_data.invocation_events:
        text = get_text_from_content(event.content)
        if text:
          parts.append(text)
    # Then fetch the final response text and append it to the end.
    final_text = get_text_from_content(content.final_response)
    if final_text:
      parts.append(final_text)

    return "\n".join(parts) if parts else None

  if content and content.parts:
    return "\n".join([p.text for p in content.parts if p.text])


def get_eval_status(score: Optional[float], threshold: float) -> EvalStatus:
  if score is None:
    return EvalStatus.NOT_EVALUATED
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED


def get_average_rubric_score(
    rubric_scores: list[RubricScore],
) -> Optional[float]:
  """Returns a single score value from the given list of rubric scores.

  It is possible that none of the rubric score actually contain a score value,
  if that happens then None is returned.

  If non-zero score values are present, then a mean value is returned as the
  aggregated value.
  """
  rubric_scores = [
      rubric_score.score
      for rubric_score in rubric_scores
      if rubric_score.score is not None
  ]

  return statistics.mean(rubric_scores) if rubric_scores else None


class _ToolDeclarations(EvalBaseModel):
  """Internal data model used for serializing Tool declarations."""

  tool_declarations: dict[str, list[Any]]


def get_tool_declarations_as_json_str(
    app_details: AppDetails,
) -> str:
  """Returns a JSON string representation of Tool declarations.

  The output of this method is usually intended to be sent to the LLM.
  """
  tool_declarations = _ToolDeclarations(
      tool_declarations=app_details.get_tools_by_agent_name()
  )
  return tool_declarations.model_dump_json(
      indent=2,
      exclude_unset=True,
      exclude_defaults=True,
      exclude_none=True,
  )


class _ToolCallAndResponse(EvalBaseModel):
  """Internal data model to capture one single tool call and response."""

  step: int
  tool_call: genai_types.FunctionCall
  tool_response: Union[genai_types.FunctionResponse, str]


class _ToolCallsAndResponses(EvalBaseModel):
  """Internal data model used for serializing Tool call and responses."""

  tool_calls_and_response: list[_ToolCallAndResponse]


def get_tool_calls_and_responses_as_json_str(
    intermediate_data: Optional[IntermediateDataType],
) -> str:
  """Returns a JSON string representation of tool calls and corresponding responses.

  The output of this method is usually intended to be sent to the LLM.
  """
  raw_tool_calls_and_response = get_all_tool_calls_with_responses(
      intermediate_data
  )

  if not raw_tool_calls_and_response:
    return "No intermediate steps were taken."

  tool_calls_and_responses = []
  for idx, (tool_call, tool_response) in enumerate(raw_tool_calls_and_response):
    tool_calls_and_responses.append(
        _ToolCallAndResponse(
            step=idx,
            tool_call=tool_call,
            tool_response=tool_response if tool_response else "None",
        )
    )

  internal_tool_calls_and_responses = _ToolCallsAndResponses(
      tool_calls_and_response=tool_calls_and_responses
  )

  return internal_tool_calls_and_responses.model_dump_json(
      indent=2,
      exclude_unset=True,
      exclude_defaults=True,
      exclude_none=True,
  )
