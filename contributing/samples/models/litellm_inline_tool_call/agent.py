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

import datetime
import json
import re
from typing import Any
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLLMClient


class InlineJsonToolClient(LiteLLMClient):
  """LiteLLM client that emits inline JSON tool calls for testing."""

  async def acompletion(self, model, messages, tools, **kwargs):
    del tools, kwargs  # Only needed for API parity.

    tool_message = _find_last_role(messages, role="tool")
    if tool_message:
      tool_summary = _coerce_to_text(tool_message.get("content"))
      return {
          "id": "mock-inline-tool-final-response",
          "model": model,
          "choices": [{
              "message": {
                  "role": "assistant",
                  "content": (
                      f"The instrumentation tool responded with: {tool_summary}"
                  ),
              },
              "finish_reason": "stop",
          }],
          "usage": {
              "prompt_tokens": 60,
              "completion_tokens": 12,
              "total_tokens": 72,
          },
      }

    timezone = _extract_timezone(messages) or "Asia/Taipei"
    inline_call = json.dumps(
        {
            "name": "get_current_time",
            "arguments": {"timezone_str": timezone},
        },
        separators=(",", ":"),
    )

    return {
        "id": "mock-inline-tool-call",
        "model": model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": (
                    f"{inline_call}\nLet me double-check the clock for you."
                ),
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 45,
            "completion_tokens": 15,
            "total_tokens": 60,
        },
    }


def _find_last_role(
    messages: list[dict[str, Any]], role: str
) -> dict[str, Any]:
  """Returns the last message with the given role."""
  for message in reversed(messages):
    if message.get("role") == role:
      return message
  return {}


def _coerce_to_text(content: Any) -> str:
  """Best-effort conversion from OpenAI message content to text."""
  if isinstance(content, str):
    return content
  if isinstance(content, dict):
    return _coerce_to_text(content.get("text"))
  if isinstance(content, list):
    texts = []
    for part in content:
      if isinstance(part, dict):
        texts.append(part.get("text") or "")
      elif isinstance(part, str):
        texts.append(part)
    return " ".join(text for text in texts if text)
  return ""


_TIMEZONE_PATTERN = re.compile(r"([A-Za-z]+/[A-Za-z_]+)")


def _extract_timezone(messages: list[dict[str, Any]]) -> str | None:
  """Extracts an IANA timezone string from the last user message."""
  user_message = _find_last_role(messages, role="user")
  text = _coerce_to_text(user_message.get("content"))
  if not text:
    return None
  match = _TIMEZONE_PATTERN.search(text)
  if match:
    return match.group(1)
  lowered = text.lower()
  if "taipei" in lowered:
    return "Asia/Taipei"
  if "new york" in lowered:
    return "America/New_York"
  if "london" in lowered:
    return "Europe/London"
  if "tokyo" in lowered:
    return "Asia/Tokyo"
  return None


def get_current_time(timezone_str: str) -> dict[str, str]:
  """Returns mock current time for the provided timezone."""
  try:
    tz = ZoneInfo(timezone_str)
  except ZoneInfoNotFoundError as exc:
    return {
        "status": "error",
        "report": f"Unable to parse timezone '{timezone_str}': {exc}",
    }
  now = datetime.datetime.now(tz)
  return {
      "status": "success",
      "report": (
          f"The current time in {timezone_str} is"
          f" {now.strftime('%Y-%m-%d %H:%M:%S %Z')}."
      ),
  }


_mock_model = LiteLlm(
    model="mock/inline-json-tool-calls",
    llm_client=InlineJsonToolClient(),
)

root_agent = Agent(
    name="litellm_inline_tool_tester",
    model=_mock_model,
    description=(
        "Demonstrates LiteLLM inline JSON tool-call parsing without an external"
        " VLLM deployment."
    ),
    instruction=(
        "You are a deterministic clock assistant. Always call the"
        " get_current_time tool before answering user questions. After the tool"
        " responds, summarize what it returned."
    ),
    tools=[get_current_time],
)
