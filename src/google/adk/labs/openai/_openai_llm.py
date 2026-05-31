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

"""OpenAI integration for GPT models."""

from __future__ import annotations

import copy
from functools import cached_property
import json
import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import Iterable
from typing import Literal
from typing import Union

from google.genai import types

try:
  from openai import AsyncOpenAI
  from openai.types.chat import ChatCompletion
  from openai.types.chat import ChatCompletionChunk
  from openai.types.chat import ChatCompletionContentPartImageParam
  from openai.types.chat import ChatCompletionMessage
  from openai.types.chat import ChatCompletionMessageParam
  from openai.types.chat import ChatCompletionToolParam
except ImportError as e:
  raise ImportError(
      "The 'openai' package is not installed. Please install it with "
      "`pip install openai` to use the OpenAILlm."
  ) from e

from pydantic import BaseModel
from typing_extensions import override

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)

__all__ = ["OpenAILlm"]


def _to_openai_role(
    role: str | None,
) -> Literal["system", "user", "assistant", "tool"]:
  if role in ["model", "assistant"]:
    return "assistant"
  if role == "system":
    return "system"
  if role == "tool":
    return "tool"
  return "user"


def _part_to_openai_content(
    part: types.Part,
) -> str | ChatCompletionContentPartImageParam:
  """Converts a genai Part to OpenAI content."""
  if part.thought and part.text:
    return f"Thought: {part.text}"
  if part.text:
    return part.text

  if part.inline_data:
    import base64

    mime_type = part.inline_data.mime_type
    data = part.inline_data.data
    if isinstance(data, bytes):
      encoded = base64.b64encode(data).decode("utf-8")
    else:
      encoded = str(data)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }

  if part.file_data:
    if part.file_data.file_uri and part.file_data.file_uri.startswith("http"):
      return {
          "type": "image_url",
          "image_url": {"url": part.file_data.file_uri},
      }

  return ""


def _content_to_openai_messages(
    content: types.Content,
) -> list[ChatCompletionMessageParam]:
  """Converts a types.Content to a list of OpenAI messages."""
  messages = []
  role = _to_openai_role(content.role)

  tool_calls = []
  content_parts = []

  for part in content.parts or []:
    if part.function_call:
      tool_calls.append({
          "id": part.function_call.id or "",
          "type": "function",
          "function": {
              "name": part.function_call.name,
              "arguments": (
                  json.dumps(part.function_call.args)
                  if part.function_call.args
                  else "{}"
              ),
          },
      })
    elif part.function_response:
      messages.append({
          "role": "tool",
          "tool_call_id": part.function_response.id or "",
          "content": (
              json.dumps(part.function_response.response)
              if part.function_response.response is not None
              else ""
          ),
      })
    else:
      content_parts.append(_part_to_openai_content(part))

  processed_parts = []
  for c in content_parts:
    if isinstance(c, str) and c:
      processed_parts.append({"type": "text", "text": c})
    elif isinstance(c, dict):
      processed_parts.append(c)

  has_images = any(p.get("type") == "image_url" for p in processed_parts)

  if not has_images:
    content_val = "\n".join(
        [p["text"] for p in processed_parts if p["type"] == "text"]
    )
  else:
    content_val = processed_parts

  if role == "assistant" and (content_val or tool_calls):
    msg = {"role": "assistant"}
    if content_val:
      msg["content"] = content_val
    if tool_calls:
      msg["tool_calls"] = tool_calls
    messages.append(msg)
  elif role == "user" and content_val:
    messages.append({
        "role": "user",
        "content": content_val,
    })
  elif role == "system" and content_val:
    if isinstance(content_val, list):
      text_only = "\n".join(
          [p["text"] for p in content_val if p["type"] == "text"]
      )
      messages.append({
          "role": "system",
          "content": text_only,
      })
    else:
      messages.append({
          "role": "system",
          "content": content_val,
      })

  return messages


def _enforce_strict_openai_schema(schema: dict[str, Any]) -> None:
  """Recursively transforms a JSON schema for OpenAI strict structured outputs."""
  if not isinstance(schema, dict):
    return
  if "$ref" in schema:
    for key in list(schema.keys()):
      if key != "$ref":
        del schema[key]
    return
  if schema.get("type") == "object" and "properties" in schema:
    schema["additionalProperties"] = False
    schema["required"] = sorted(schema["properties"].keys())
  for defn in schema.get("$defs", {}).values():
    _enforce_strict_openai_schema(defn)
  for prop in schema.get("properties", {}).values():
    _enforce_strict_openai_schema(prop)
  for key in ("anyOf", "oneOf", "allOf"):
    for item in schema.get(key, []):
      _enforce_strict_openai_schema(item)
  if "items" in schema and isinstance(schema["items"], dict):
    _enforce_strict_openai_schema(schema["items"])


def _update_type_string(value: Any):
  """Lowercases nested JSON schema type strings for OpenAI compatibility."""
  if isinstance(value, list):
    for item in value:
      _update_type_string(item)
    return

  if not isinstance(value, dict):
    return

  schema_type = value.get("type")
  if isinstance(schema_type, str):
    value["type"] = schema_type.lower()

  for dict_key in (
      "$defs",
      "defs",
      "dependentSchemas",
      "patternProperties",
      "properties",
  ):
    child_dict = value.get(dict_key)
    if isinstance(child_dict, dict):
      for child_value in child_dict.values():
        _update_type_string(child_value)

  for single_key in (
      "additionalProperties",
      "additional_properties",
      "contains",
      "else",
      "if",
      "items",
      "not",
      "propertyNames",
      "then",
      "unevaluatedProperties",
  ):
    child_value = value.get(single_key)
    if isinstance(child_value, (dict, list)):
      _update_type_string(child_value)

  for list_key in (
      "allOf",
      "all_of",
      "anyOf",
      "any_of",
      "oneOf",
      "one_of",
      "prefixItems",
  ):
    child_list = value.get(list_key)
    if isinstance(child_list, list):
      _update_type_string(child_list)


def _function_declaration_to_openai_tool(
    function_declaration: types.FunctionDeclaration,
) -> ChatCompletionToolParam:
  """Converts a function declaration to an OpenAI tool param."""
  if not function_declaration.name:
    raise ValueError("FunctionDeclaration must have a name.")

  # Use parameters_json_schema if available, otherwise convert from parameters
  if function_declaration.parameters_json_schema:
    parameters = copy.deepcopy(function_declaration.parameters_json_schema)
    _update_type_string(parameters)
  else:
    properties = {}
    required_params = []
    if function_declaration.parameters:
      if function_declaration.parameters.properties:
        for key, value in function_declaration.parameters.properties.items():
          properties[key] = value.model_dump(by_alias=True, exclude_none=True)
      if function_declaration.parameters.required:
        required_params = function_declaration.parameters.required

    parameters = {
        "type": "object",
        "properties": properties,
    }
    if required_params:
      parameters["required"] = required_params
    _update_type_string(parameters)

  return {
      "type": "function",
      "function": {
          "name": function_declaration.name,
          "description": function_declaration.description or "",
          "parameters": parameters,
      },
  }


def _response_to_llm_response(response: ChatCompletion) -> LlmResponse:
  """Parses an OpenAI response into an LlmResponse."""
  choice = response.choices[0]
  message = choice.message

  parts = []
  if message.content:
    parts.append(types.Part.from_text(text=message.content))

  if message.tool_calls:
    for tool_call in message.tool_calls:
      args = {}
      if tool_call.function.arguments:
        try:
          args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
          logger.warning("Failed to parse tool call arguments as JSON.")

      part = types.Part.from_function_call(
          name=tool_call.function.name, args=args
      )
      part.function_call.id = tool_call.id
      parts.append(part)

  return LlmResponse(
      content=types.Content(
          role="model",
          parts=parts,
      ),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=response.usage.prompt_tokens,
          candidates_token_count=response.usage.completion_tokens,
          total_token_count=response.usage.total_tokens,
      ),
  )


class OpenAILlm(BaseLlm):
  """Integration with OpenAI models.

  Attributes:
      model: The name of the OpenAI model.
      max_tokens: The maximum number of tokens to generate.
  """

  model: str = "gpt-4o"
  max_tokens: int = 4096

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r"gpt-.*", r"o1-.*", r"o3-.*"]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    messages = []
    if llm_request.config and llm_request.config.system_instruction:
      messages.append({
          "role": "system",
          "content": llm_request.config.system_instruction,
      })

    for content in llm_request.contents or []:
      messages.extend(_content_to_openai_messages(content))

    tools = []
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
      tools = [
          _function_declaration_to_openai_tool(tool)
          for tool in llm_request.config.tools[0].function_declarations
      ]

    tool_choice = "auto" if tools else None

    response_format = None
    if llm_request.config and llm_request.config.response_schema:
      schema = llm_request.config.response_schema
      schema_name = "response"
      schema_dict = {}

      if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_dict = schema.model_json_schema()
        schema_name = schema.__name__
      elif isinstance(schema, BaseModel):
        schema_dict = schema.__class__.model_json_schema()
        schema_name = schema.__class__.__name__
      elif isinstance(schema, dict):
        schema_dict = copy.deepcopy(schema)
        if "title" in schema_dict:
          schema_name = str(schema_dict["title"])

      if schema_dict:
        _enforce_strict_openai_schema(schema_dict)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema_dict,
            },
        }
    elif (
        llm_request.config
        and llm_request.config.response_mime_type == "application/json"
    ):
      response_format = {"type": "json_object"}

    kwargs = {
        "model": self.model,
        "messages": messages,
        "tools": tools if tools else None,
        "tool_choice": tool_choice,
        "max_tokens": self.max_tokens,
        "response_format": response_format,
    }

    if llm_request.config:
      if getattr(llm_request.config, "temperature", None) is not None:
        kwargs["temperature"] = llm_request.config.temperature
      if getattr(llm_request.config, "top_p", None) is not None:
        kwargs["top_p"] = llm_request.config.top_p
      if getattr(llm_request.config, "stop_sequences", None):
        kwargs["stop"] = llm_request.config.stop_sequences
      if getattr(llm_request.config, "max_output_tokens", None) is not None:
        kwargs["max_tokens"] = llm_request.config.max_output_tokens

    if not stream:
      response = await self._openai_client.chat.completions.create(**kwargs)
      yield _response_to_llm_response(response)
    else:
      async for response in self._generate_content_streaming(kwargs):
        yield response

  async def _generate_content_streaming(
      self,
      kwargs: dict[str, Any],
  ) -> AsyncGenerator[LlmResponse, None]:
    """Handles streaming responses from OpenAI models."""
    kwargs["stream"] = True
    raw_stream = await self._openai_client.chat.completions.create(**kwargs)

    text_accumulated = ""
    tool_calls_accumulated: dict[int, dict[str, Any]] = {}

    async for chunk in raw_stream:
      if not chunk.choices:
        continue
      choice = chunk.choices[0]
      delta = choice.delta

      if delta.content:
        text_accumulated += delta.content
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=delta.content)],
            ),
            partial=True,
        )

      if delta.tool_calls:
        for tc_delta in delta.tool_calls:
          index = tc_delta.index
          if index not in tool_calls_accumulated:
            tool_calls_accumulated[index] = {
                "id": tc_delta.id,
                "name": tc_delta.function.name,
                "arguments": "",
            }
          if tc_delta.function.arguments:
            tool_calls_accumulated[index][
                "arguments"
            ] += tc_delta.function.arguments

    # Yield final response with all accumulated content
    parts = []
    if text_accumulated:
      parts.append(types.Part.from_text(text=text_accumulated))

    for index in sorted(tool_calls_accumulated.keys()):
      acc = tool_calls_accumulated[index]
      args = {}
      if acc["arguments"]:
        try:
          args = json.loads(acc["arguments"])
        except json.JSONDecodeError:
          logger.warning(
              "Failed to parse accumulated tool call arguments as JSON."
          )

      part = types.Part.from_function_call(name=acc["name"], args=args)
      part.function_call.id = acc["id"]
      parts.append(part)

    yield LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=False,
    )

  @cached_property
  def _openai_client(self) -> AsyncOpenAI:
    return AsyncOpenAI()
