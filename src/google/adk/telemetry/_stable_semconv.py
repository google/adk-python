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

"""Helpers for building log bodies that follow the stable OTel GenAI semconv.

This module centralizes the construction of `gen_ai.system.message`,
`gen_ai.user.message`, and `gen_ai.choice` log bodies so that both the
tracing layer (which emits the logs) and the ADK Web UI exporter (which
rebuilds the bodies after elision) share the same shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from opentelemetry.util.types import AnyValue

from ._serialization import serialize_content
from .context import TelemetryConfig

if TYPE_CHECKING:
  from google.genai import types

  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse

# Stable OTel GenAI semantic-convention event names.
GEN_AI_SYSTEM_MESSAGE_EVENT = "gen_ai.system.message"
GEN_AI_USER_MESSAGE_EVENT = "gen_ai.user.message"
GEN_AI_CHOICE_EVENT = "gen_ai.choice"

# Standard OTEL env variable that controls whether prompt/response content is
# included in log bodies. When unset/false, content is replaced with <elided>.
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)

USER_CONTENT_ELIDED = "<elided>"


def _serialize_content_with_optional_elision(
    content: types.ContentUnion | None, *, capture_content: bool
) -> AnyValue:
  if not capture_content:
    return USER_CONTENT_ELIDED
  if content is None:
    return None
  return serialize_content(content)


def system_message_body(
    llm_request: LlmRequest,
    telemetry_config: TelemetryConfig,
    *,
    do_not_elide: bool = False,
) -> Mapping[str, AnyValue]:
  """Builds the body for a `gen_ai.system.message` log event.

  Args:
    llm_request: The LLM request whose system instruction should be logged.
    do_not_elide_content: When True, always include the content regardless of
      the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env var. The Web
      UI exporter sets this to True because the UI needs the full content.
  """
  system_instruction = None
  if llm_request.config is not None:
    system_instruction = llm_request.config.system_instruction
  return {
      "content": _serialize_content_with_optional_elision(
          system_instruction,
          capture_content=do_not_elide
          or telemetry_config.should_add_content_to_logs,
      )
  }


def user_message_body(
    content: types.ContentUnion | None,
    telemetry_config: TelemetryConfig,
    *,
    do_not_elide: bool = False,
) -> Mapping[str, AnyValue]:
  """Builds the body for a single `gen_ai.user.message` log event.

  Args:
    content: The user content for this message. Callers that emit multiple user
      messages (e.g. tracing's per-content loop) call this builder once per
      content.
    do_not_elide_content: When True, always include the content regardless of
      the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env var.
  """
  return {
      "content": _serialize_content_with_optional_elision(
          content,
          capture_content=do_not_elide
          or telemetry_config.should_add_content_to_logs,
      )
  }


def choice_body(
    llm_response: LlmResponse | None,
    telemetry_config: TelemetryConfig,
    *,
    do_not_elide: bool = False,
) -> Mapping[str, AnyValue]:
  """Builds the body for a `gen_ai.choice` log event.

  ADK always returns a single candidate, so `index` is always 0.
  `finish_reason` is included only when present on the response.

  Args:
    llm_response: The LLM response describing the choice.
    do_not_elide_content: When True, always include the content regardless of
      the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env var.
  """
  if llm_response is None:
    return {"content": None, "index": 0}
  body: dict[str, AnyValue] = {
      "content": _serialize_content_with_optional_elision(
          llm_response.content,
          capture_content=do_not_elide
          or telemetry_config.should_add_content_to_logs,
      ),
      "index": 0,  # ADK always returns a single candidate.
  }
  if llm_response.finish_reason is not None:
    finish_reason = llm_response.finish_reason
    body["finish_reason"] = (
        finish_reason.value
        if hasattr(finish_reason, "value")
        else str(finish_reason)
    )
  return body
