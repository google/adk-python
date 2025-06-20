# Copyright 2025 Google LLC
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

"""Anthropic integration for Claude models."""

from __future__ import annotations

from functools import cached_property
import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import Generator
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from anthropic import AnthropicVertex
from anthropic import NOT_GIVEN
from anthropic import types as anthropic_types
from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

__all__ = ["Claude"]

logger = logging.getLogger("google_adk." + __name__)

MAX_TOKEN = 1024


class ClaudeRequest(BaseModel):
  system_instruction: str
  messages: Iterable[anthropic_types.MessageParam]
  tools: list[anthropic_types.ToolParam]


def to_claude_role(role: Optional[str]) -> Literal["user", "assistant"]:
  if role in ["model", "assistant"]:
    return "assistant"
  return "user"


def to_google_genai_finish_reason(
    anthropic_stop_reason: Optional[str],
) -> types.FinishReason:
  if anthropic_stop_reason in ["end_turn", "stop_sequence", "tool_use"]:
    return "STOP"
  if anthropic_stop_reason == "max_tokens":
    return "MAX_TOKENS"
  return "FINISH_REASON_UNSPECIFIED"


def part_to_message_block(
    part: types.Part,
) -> Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.ToolUseBlockParam,
    anthropic_types.ToolResultBlockParam,
]:
  if part.text:
    return anthropic_types.TextBlockParam(text=part.text, type="text")
  if part.function_call:
    assert part.function_call.name

    return anthropic_types.ToolUseBlockParam(
        id=part.function_call.id or "",
        name=part.function_call.name,
        input=part.function_call.args,
        type="tool_use",
    )
  if part.function_response:
    content = ""
    if (
        "result" in part.function_response.response
        and part.function_response.response["result"]
    ):
      # Transformation is required because the content is a list of dict.
      # ToolResultBlockParam content doesn't support list of dict. Converting
      # to str to prevent anthropic.BadRequestError from being thrown.
      content = str(part.function_response.response["result"])
    return anthropic_types.ToolResultBlockParam(
        tool_use_id=part.function_response.id or "",
        type="tool_result",
        content=content,
        is_error=False,
    )
  raise NotImplementedError("Not supported yet.")


def content_to_message_param(
    content: types.Content,
) -> anthropic_types.MessageParam:
  return {
      "role": to_claude_role(content.role),
      "content": [part_to_message_block(part) for part in content.parts or []],
  }


def content_block_to_part(
    content_block: anthropic_types.ContentBlock,
) -> types.Part:
  if isinstance(content_block, anthropic_types.TextBlock):
    return types.Part.from_text(text=content_block.text)
  if isinstance(content_block, anthropic_types.ToolUseBlock):
    assert isinstance(content_block.input, dict)
    part = types.Part.from_function_call(
        name=content_block.name, args=content_block.input
    )
    part.function_call.id = content_block.id
    return part
  raise NotImplementedError("Not supported yet.")


def message_to_generate_content_response(
    message: anthropic_types.Message,
) -> LlmResponse:
  logger.info(
      "Claude response: %s",
      message.model_dump_json(indent=2, exclude_none=True),
  )

  return LlmResponse(
      content=types.Content(
          role="model",
          parts=[content_block_to_part(cb) for cb in message.content],
      ),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=message.usage.input_tokens,
          candidates_token_count=message.usage.output_tokens,
          total_token_count=(
              message.usage.input_tokens + message.usage.output_tokens
          ),
      ),
      # TODO: Deal with these later.
      # finish_reason=to_google_genai_finish_reason(message.stop_reason),
  )


def _update_type_string(value_dict: dict[str, Any]):
  """Updates 'type' field to expected JSON schema format."""
  if "type" in value_dict:
    value_dict["type"] = value_dict["type"].lower()

  if "items" in value_dict:
    # 'type' field could exist for items as well, this would be the case if
    # items represent primitive types.
    _update_type_string(value_dict["items"])

    if "properties" in value_dict["items"]:
      # There could be properties as well on the items, especially if the items
      # are complex object themselves. We recursively traverse each individual
      # property as well and fix the "type" value.
      for _, value in value_dict["items"]["properties"].items():
        _update_type_string(value)


def function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> anthropic_types.ToolParam:
  assert function_declaration.name

  properties = {}
  if (
      function_declaration.parameters
      and function_declaration.parameters.properties
  ):
    for key, value in function_declaration.parameters.properties.items():
      value_dict = value.model_dump(exclude_none=True)
      _update_type_string(value_dict)
      properties[key] = value_dict

  return anthropic_types.ToolParam(
      name=function_declaration.name,
      description=function_declaration.description or "",
      input_schema={
          "type": "object",
          "properties": properties,
      },
  )


class Claude(BaseLlm):
  """Integration with Claude models served from Vertex AI.

  Attributes:
    model: The name of the Claude model.
  """

  model: str = "claude-3-5-sonnet-v2@20241022"

  @staticmethod
  @override
  def supported_models() -> list[str]:
    return [r"claude-3-.*", r"claude-.*-4.*"]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    messages = [
        content_to_message_param(content)
        for content in llm_request.contents or []
    ]
    tools = NOT_GIVEN
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
      tools = [
          function_declaration_to_tool_param(tool)
          for tool in llm_request.config.tools[0].function_declarations
      ]
    tool_choice = (
        anthropic_types.ToolChoiceAutoParam(type="auto")
        if llm_request.tools_dict
        else NOT_GIVEN
    )
    # TODO(b/421255973): Enable streaming for anthropic models.
    message = self._anthropic_client.messages.create(
        model=llm_request.model,
        system=llm_request.config.system_instruction,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=MAX_TOKEN,
    )
    yield message_to_generate_content_response(message)

  @cached_property
  def _anthropic_client(self) -> AnthropicVertex:
    if (
        "GOOGLE_CLOUD_PROJECT" not in os.environ
        or "GOOGLE_CLOUD_LOCATION" not in os.environ
    ):
      raise ValueError(
          "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for using"
          " Anthropic on Vertex."
      )

    return AnthropicVertex(
        project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
        region=os.environ["GOOGLE_CLOUD_LOCATION"],
    )
