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

"""Helpers for the labs OpenAI wrappers.

Home for parsing/mapping logic that the Chat Completions wrapper
(``_openai_llm.py``) and the Responses wrapper (``_openai_responses_llm.py``)
can share. Currently this is the Chat Completions finish-reason mapper; the
Responses model keeps its own status-based ``_map_finish_reason`` (it maps a
Responses API *status*, not a finish-reason string), so that mapper is not
shared here.
"""

from __future__ import annotations

from google.genai import types

__all__ = [
    "map_finish_reason",
]


def map_finish_reason(
    finish_reason: str | None,
) -> types.FinishReason | None:
  """Maps an OpenAI chat-completion finish reason to an ADK FinishReason.

  A finish reason that is present but not one we recognize maps to
  ``FINISH_REASON_UNSPECIFIED``, matching the convention in
  ``models/anthropic_llm.py`` and ``models/apigee_llm.py`` (``OTHER`` is
  reserved for recognized abnormal terminations).
  """
  if finish_reason in ("stop", "tool_calls", "function_call"):
    return types.FinishReason.STOP
  if finish_reason == "length":
    return types.FinishReason.MAX_TOKENS
  if finish_reason == "content_filter":
    return types.FinishReason.SAFETY
  if not finish_reason:
    return None
  return types.FinishReason.FINISH_REASON_UNSPECIFIED
