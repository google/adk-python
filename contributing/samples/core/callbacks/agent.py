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

from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import BaseTool
from google.adk.tools import ToolContext
from google.genai import types


def get_weather(city: str) -> str:
  return f"The weather in {city} is sunny."


def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
  """A callback that runs before a tool is called.

  Args:
      tool: The tool instance being called.
      args: The arguments passed to the tool.
      tool_context: The context for the tool execution.

  Returns:
      A dict containing the mock response if the call should be short-circuited,
      or None to proceed with the actual tool call.
  """
  # Intercept tool calls for London and return a mocked response
  if args.get("city") == "London":
    return {
        "result": "Weather in London is always rainy (intercepted by callback)."
    }

  return None


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
  """A callback that runs before the model is called.

  Args:
      callback_context: The context for the callback.
      llm_request: The request that is about to be sent to the model.

  Returns:
      An LlmResponse to short-circuit the model call, or None to proceed.
  """
  # Short-circuit if the user simply says "Hi"
  if llm_request.contents:
    last_content = llm_request.contents[-1]
    if last_content.parts:
      last_part = last_content.parts[-1]
      if last_part.text and last_part.text.strip().lower() == "hi":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Hello from before_model callback!"
                    )
                ],
            )
        )

  return None


def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
  """A callback that runs after the model is called."""
  if llm_response.usage_metadata:
    usage = llm_response.usage_metadata
    usage_text = (
        "\n\nafter_model_callback: [Token Usage:"
        f" Input={usage.prompt_token_count},"
        f" Output={usage.candidates_token_count}]"
    )

    if not llm_response.content:
      llm_response.content = types.Content(role="model", parts=[])

    llm_response.content.parts.append(types.Part.from_text(text=usage_text))
    print(llm_response.content)

  return llm_response


root_agent = Agent(
    name="callback_demo_agent",
    tools=[get_weather],
    before_tool_callback=before_tool_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)
