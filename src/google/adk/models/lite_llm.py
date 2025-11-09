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

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import Union
import warnings

from google.genai import types
import litellm
from litellm import acompletion
from litellm import ChatCompletionAssistantMessage
from litellm import ChatCompletionAssistantToolCall
from litellm import ChatCompletionDeveloperMessage
from litellm import ChatCompletionMessageToolCall
from litellm import ChatCompletionToolMessage
from litellm import ChatCompletionUserMessage
from litellm import completion
from litellm import CustomStreamWrapper
from litellm import Function
from litellm import Message
from litellm import ModelResponse
from litellm import OpenAIMessageContent
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

# This will add functions to prompts if functions are provided.
litellm.add_function_to_prompt = True

logger = logging.getLogger("google_adk." + __name__)

_NEW_LINE = "\n"
_EXCLUDED_PART_FIELD = {"inline_data": {"data"}}
_LITELLM_STRUCTURED_TYPES = {"json_object", "json_schema"}

# Mapping of LiteLLM finish_reason strings to FinishReason enum values
# Note: tool_calls/function_call map to STOP because:
# 1. FinishReason.TOOL_CALL enum does not exist (as of google-genai 0.8.0)
# 2. Tool calls represent normal completion (model stopped to invoke tools)
# 3. Gemini native responses use STOP for tool calls (see lite_llm.py:910)
_FINISH_REASON_MAPPING = {
    "length": types.FinishReason.MAX_TOKENS,
    "stop": types.FinishReason.STOP,
    "tool_calls": (
        types.FinishReason.STOP
    ),  # Normal completion with tool invocation
    "function_call": types.FinishReason.STOP,  # Legacy function call variant
    "content_filter": types.FinishReason.SAFETY,
}

_SUPPORTED_FILE_CONTENT_MIME_TYPES = set(
    ["application/pdf", "application/json", "text/plain"]
)


class ChatCompletionFileUrlObject(TypedDict, total=False):
  file_data: str
  file_id: str
  format: str


class FunctionChunk(BaseModel):
  id: Optional[str]
  name: Optional[str]
  args: Optional[str]
  index: Optional[int] = 0


class TextChunk(BaseModel):
  text: str


class UsageMetadataChunk(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  cached_prompt_tokens: int = 0


class LiteLLMClient:
  """Provides acompletion method (for better testability)."""

  async def acompletion(
      self, model, messages, tools, **kwargs
  ) -> Union[ModelResponse, CustomStreamWrapper]:
    """Asynchronously calls acompletion.

    Args:
      model: The model name.
      messages: The messages to send to the model.
      tools: The tools to use for the model.
      **kwargs: Additional arguments to pass to acompletion.

    Returns:
      The model response as a message.
    """

    return await acompletion(
        model=model,
        messages=messages,
        tools=tools,
        **kwargs,
    )

  def completion(
      self, model, messages, tools, stream=False, **kwargs
  ) -> Union[ModelResponse, CustomStreamWrapper]:
    """Synchronously calls completion. This is used for streaming only.

    Args:
      model: The model to use.
      messages: The messages to send.
      tools: The tools to use for the model.
      stream: Whether to stream the response.
      **kwargs: Additional arguments to pass to completion.

    Returns:
      The response from the model.
    """

    return completion(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        **kwargs,
    )


def _safe_json_serialize(obj) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or string.
  """

  try:
    # Try direct JSON serialization first
    return json.dumps(obj, ensure_ascii=False)
  except (TypeError, OverflowError):
    return str(obj)


def _part_has_payload(part: types.Part) -> bool:
  """Checks whether a Part contains usable payload for the model."""
  if part.text:
    return True
  if part.inline_data and part.inline_data.data:
    return True
  if part.file_data and (part.file_data.file_uri or part.file_data.data):
    return True
  return False


def _append_fallback_user_content_if_missing(
    llm_request: LlmRequest,
) -> None:
  """Ensures there is a user message with content for LiteLLM backends.

  Args:
    llm_request: The request that may need a fallback user message.
  """
  for content in reversed(llm_request.contents):
    if content.role == "user":
      parts = content.parts or []
      if any(_part_has_payload(part) for part in parts):
        return
      if not parts:
        content.parts = []
      content.parts.append(
          types.Part.from_text(
              text="Handle the requests as specified in the System Instruction."
          )
      )
      return
  llm_request.contents.append(
      types.Content(
          role="user",
          parts=[
              types.Part.from_text(
                  text=(
                      "Handle the requests as specified in the System"
                      " Instruction."
                  )
              ),
          ],
      )
  )


def _extract_cached_prompt_tokens(usage: Any) -> int:
  """Extracts cached prompt tokens from LiteLLM usage.

  Providers expose cached token metrics in different shapes. Common patterns:
  - usage["prompt_tokens_details"]["cached_tokens"] (OpenAI/Azure style)
  - usage["prompt_tokens_details"] is a list of dicts with cached_tokens
  - usage["cached_prompt_tokens"] (LiteLLM-normalized for some providers)
  - usage["cached_tokens"] (flat)

  Args:
    usage: Usage dictionary from LiteLLM response.

  Returns:
    Integer number of cached prompt tokens if present; otherwise 0.
  """
  try:
    usage_dict = usage
    if hasattr(usage, "model_dump"):
      usage_dict = usage.model_dump()
    elif isinstance(usage, str):
      try:
        usage_dict = json.loads(usage)
      except json.JSONDecodeError:
        return 0

    if not isinstance(usage_dict, dict):
      return 0

    details = usage_dict.get("prompt_tokens_details")
    if isinstance(details, dict):
      value = details.get("cached_tokens")
      if isinstance(value, int):
        return value
    elif isinstance(details, list):
      total = sum(
          item.get("cached_tokens", 0)
          for item in details
          if isinstance(item, dict)
          and isinstance(item.get("cached_tokens"), int)
      )
      if total > 0:
        return total

    for key in ("cached_prompt_tokens", "cached_tokens"):
      value = usage_dict.get(key)
      if isinstance(value, int):
        return value
  except (TypeError, AttributeError) as e:
    logger.debug("Error extracting cached prompt tokens: %s", e)

  return 0


def _content_to_message_param(
    content: types.Content,
) -> Union[Message, list[Message]]:
  """Converts a types.Content to a litellm Message or list of Messages.

  Handles multipart function responses by returning a list of
  ChatCompletionToolMessage objects if multiple function_response parts exist.

  Args:
    content: The content to convert.

  Returns:
    A litellm Message, a list of litellm Messages.
  """

  tool_messages = []
  for part in content.parts:
    if part.function_response:
      tool_messages.append(
          ChatCompletionToolMessage(
              role="tool",
              tool_call_id=part.function_response.id,
              content=_safe_json_serialize(part.function_response.response),
          )
      )
  if tool_messages:
    return tool_messages if len(tool_messages) > 1 else tool_messages[0]

  # Handle user or assistant messages
  role = _to_litellm_role(content.role)
  message_content = _get_content(content.parts) or None

  if role == "user":
    return ChatCompletionUserMessage(role="user", content=message_content)
  else:  # assistant/model
    tool_calls = []
    content_present = False
    for part in content.parts:
      if part.function_call:
        tool_calls.append(
            ChatCompletionAssistantToolCall(
                type="function",
                id=part.function_call.id,
                function=Function(
                    name=part.function_call.name,
                    arguments=_safe_json_serialize(part.function_call.args),
                ),
            )
        )
      elif part.text or part.inline_data:
        content_present = True

    final_content = message_content if content_present else None
    if final_content and isinstance(final_content, list):
      # when the content is a single text object, we can use it directly.
      # this is needed for ollama_chat provider which fails if content is a list
      final_content = (
          final_content[0].get("text", "")
          if final_content[0].get("type", None) == "text"
          else final_content
      )

    return ChatCompletionAssistantMessage(
        role=role,
        content=final_content,
        tool_calls=tool_calls or None,
    )


def _get_content(
    parts: Iterable[types.Part],
) -> Union[OpenAIMessageContent, str]:
  """Converts a list of parts to litellm content.

  Args:
    parts: The parts to convert.

  Returns:
    The litellm content.
  """

  content_objects = []
  for part in parts:
    if part.text:
      if len(parts) == 1:
        return part.text
      content_objects.append({
          "type": "text",
          "text": part.text,
      })
    elif (
        part.inline_data
        and part.inline_data.data
        and part.inline_data.mime_type
    ):
      base64_string = base64.b64encode(part.inline_data.data).decode("utf-8")
      data_uri = f"data:{part.inline_data.mime_type};base64,{base64_string}"
      # LiteLLM providers extract the MIME type from the data URI; avoid
      # passing a separate `format` field that some backends reject.

      if part.inline_data.mime_type.startswith("image"):
        content_objects.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })
      elif part.inline_data.mime_type.startswith("video"):
        content_objects.append({
            "type": "video_url",
            "video_url": {"url": data_uri},
        })
      elif part.inline_data.mime_type.startswith("audio"):
        content_objects.append({
            "type": "audio_url",
            "audio_url": {"url": data_uri},
        })
      elif part.inline_data.mime_type in _SUPPORTED_FILE_CONTENT_MIME_TYPES:
        content_objects.append({
            "type": "file",
            "file": {"file_data": data_uri},
        })
      else:
        raise ValueError("LiteLlm(BaseLlm) does not support this content part.")
    elif part.file_data and part.file_data.file_uri:
      file_object: ChatCompletionFileUrlObject = {
          "file_id": part.file_data.file_uri,
      }
      content_objects.append({
          "type": "file",
          "file": file_object,
      })

  return content_objects


def _to_litellm_role(role: Optional[str]) -> Literal["user", "assistant"]:
  """Converts a types.Content role to a litellm role.

  Args:
    role: The types.Content role.

  Returns:
    The litellm role.
  """

  if role in ["model", "assistant"]:
    return "assistant"
  return "user"


TYPE_LABELS = {
    "STRING": "string",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "OBJECT": "object",
    "ARRAY": "array",
    "INTEGER": "integer",
}


def _schema_to_dict(schema: types.Schema) -> dict:
  """Recursively converts a types.Schema to a pure-python dict

  with all enum values written as lower-case strings.

  Args:
    schema: The schema to convert.

  Returns:
    The dictionary representation of the schema.
  """
  # Dump without json encoding so we still get Enum members
  schema_dict = schema.model_dump(exclude_none=True)

  # ---- normalise this level ------------------------------------------------
  if "type" in schema_dict:
    # schema_dict["type"] can be an Enum or a str
    t = schema_dict["type"]
    schema_dict["type"] = (t.value if isinstance(t, types.Type) else t).lower()

  # ---- recurse into `items` -----------------------------------------------
  if "items" in schema_dict:
    schema_dict["items"] = _schema_to_dict(
        schema.items
        if isinstance(schema.items, types.Schema)
        else types.Schema.model_validate(schema_dict["items"])
    )

  # ---- recurse into `properties` ------------------------------------------
  if "properties" in schema_dict:
    new_props = {}
    for key, value in schema_dict["properties"].items():
      # value is a dict → rebuild a Schema object and recurse
      if isinstance(value, dict):
        new_props[key] = _schema_to_dict(types.Schema.model_validate(value))
      # value is already a Schema instance
      elif isinstance(value, types.Schema):
        new_props[key] = _schema_to_dict(value)
      # plain dict without nested schemas
      else:
        new_props[key] = value
        if "type" in new_props[key]:
          new_props[key]["type"] = new_props[key]["type"].lower()
    schema_dict["properties"] = new_props

  return schema_dict


def _function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> dict:
  """Converts a types.FunctionDeclaration to an openapi spec dictionary.

  Args:
    function_declaration: The function declaration to convert.

  Returns:
    The openapi spec dictionary representation of the function declaration.
  """

  assert function_declaration.name

  parameters = {
      "type": "object",
      "properties": {},
  }
  if (
      function_declaration.parameters
      and function_declaration.parameters.properties
  ):
    properties = {}
    for key, value in function_declaration.parameters.properties.items():
      properties[key] = _schema_to_dict(value)

    parameters = {
        "type": "object",
        "properties": properties,
    }
  elif function_declaration.parameters_json_schema:
    parameters = function_declaration.parameters_json_schema

  tool_params = {
      "type": "function",
      "function": {
          "name": function_declaration.name,
          "description": function_declaration.description or "",
          "parameters": parameters,
      },
  }

  if (
      function_declaration.parameters
      and function_declaration.parameters.required
  ):
    tool_params["function"]["parameters"][
        "required"
    ] = function_declaration.parameters.required

  return tool_params


def _model_response_to_chunk(
    response: ModelResponse,
) -> Generator[
    Tuple[
        Optional[Union[TextChunk, FunctionChunk, UsageMetadataChunk]],
        Optional[str],
    ],
    None,
    None,
]:
  """Converts a litellm message to text, function or usage metadata chunk.

  Args:
    response: The response from the model.

  Yields:
    A tuple of text or function or usage metadata chunk and finish reason.
  """

  message = None
  if response.get("choices", None):
    message = response["choices"][0].get("message", None)
    finish_reason = response["choices"][0].get("finish_reason", None)
    # check streaming delta
    if message is None and response["choices"][0].get("delta", None):
      message = response["choices"][0]["delta"]

    if message.get("content", None):
      yield TextChunk(text=message.get("content")), finish_reason

    if message.get("tool_calls", None):
      for tool_call in message.get("tool_calls"):
        # aggregate tool_call
        if tool_call.type == "function":
          func_name = tool_call.function.name
          func_args = tool_call.function.arguments

          # Ignore empty chunks that don't carry any information.
          if not func_name and not func_args:
            continue

          yield FunctionChunk(
              id=tool_call.id,
              name=func_name,
              args=func_args,
              index=tool_call.index,
          ), finish_reason

    if finish_reason and not (
        message.get("content", None) or message.get("tool_calls", None)
    ):
      yield None, finish_reason

  if not message:
    yield None, None

  # Ideally usage would be expected with the last ModelResponseStream with a
  # finish_reason set. But this is not the case we are observing from litellm.
  # So we are sending it as a separate chunk to be set on the llm_response.
  if response.get("usage", None):
    yield UsageMetadataChunk(
        prompt_tokens=response["usage"].get("prompt_tokens", 0),
        completion_tokens=response["usage"].get("completion_tokens", 0),
        total_tokens=response["usage"].get("total_tokens", 0),
        cached_prompt_tokens=_extract_cached_prompt_tokens(response["usage"]),
    ), None


def _model_response_to_generate_content_response(
    response: ModelResponse,
) -> LlmResponse:
  """Converts a litellm response to LlmResponse. Also adds usage metadata.

  Args:
    response: The model response.

  Returns:
    The LlmResponse.
  """

  message = None
  finish_reason = None
  if (choices := response.get("choices")) and choices:
    first_choice = choices[0]
    message = first_choice.get("message", None)
    finish_reason = first_choice.get("finish_reason", None)

  if not message:
    raise ValueError("No message in response")

  llm_response = _message_to_generate_content_response(
      message, model_version=response.model
  )
  if finish_reason:
    # If LiteLLM already provides a FinishReason enum (e.g., for Gemini), use
    # it directly. Otherwise, map the finish_reason string to the enum.
    if isinstance(finish_reason, types.FinishReason):
      llm_response.finish_reason = finish_reason
    else:
      finish_reason_str = str(finish_reason).lower()
      llm_response.finish_reason = _FINISH_REASON_MAPPING.get(
          finish_reason_str, types.FinishReason.OTHER
      )
  if response.get("usage", None):
    llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=response["usage"].get("prompt_tokens", 0),
        candidates_token_count=response["usage"].get("completion_tokens", 0),
        total_token_count=response["usage"].get("total_tokens", 0),
        cached_content_token_count=_extract_cached_prompt_tokens(
            response["usage"]
        ),
    )
  return llm_response


def _message_to_generate_content_response(
    message: Message, *, is_partial: bool = False, model_version: str = None
) -> LlmResponse:
  """Converts a litellm message to LlmResponse.

  Args:
    message: The message to convert.
    is_partial: Whether the message is partial.
    model_version: The model version used to generate the response.

  Returns:
    The LlmResponse.
  """

  parts = []
  if message.get("content", None):
    parts.append(types.Part.from_text(text=message.get("content")))

  if message.get("tool_calls", None):
    for tool_call in message.get("tool_calls"):
      if tool_call.type == "function":
        part = types.Part.from_function_call(
            name=tool_call.function.name,
            args=json.loads(tool_call.function.arguments or "{}"),
        )
        part.function_call.id = tool_call.id
        parts.append(part)

  return LlmResponse(
      content=types.Content(role="model", parts=parts),
      partial=is_partial,
      model_version=model_version,
  )


def _to_litellm_response_format(
    response_schema: types.SchemaUnion,
) -> Optional[Dict[str, Any]]:
  """Converts ADK response schema objects into LiteLLM-compatible payloads."""

  if isinstance(response_schema, dict):
    schema_type = response_schema.get("type")
    if (
        isinstance(schema_type, str)
        and schema_type.lower() in _LITELLM_STRUCTURED_TYPES
    ):
      return response_schema
    schema_dict = dict(response_schema)
  elif isinstance(response_schema, type) and issubclass(
      response_schema, BaseModel
  ):
    schema_dict = response_schema.model_json_schema()
  elif isinstance(response_schema, BaseModel):
    if isinstance(response_schema, types.Schema):
      # GenAI Schema instances already represent JSON schema definitions.
      schema_dict = response_schema.model_dump(exclude_none=True, mode="json")
    else:
      schema_dict = response_schema.__class__.model_json_schema()
  elif hasattr(response_schema, "model_dump"):
    schema_dict = response_schema.model_dump(exclude_none=True, mode="json")
  else:
    logger.warning(
        "Unsupported response_schema type %s for LiteLLM structured outputs.",
        type(response_schema),
    )
    return None

  return {
      "type": "json_object",
      "response_schema": schema_dict,
  }


def _get_completion_inputs(
    llm_request: LlmRequest,
) -> Tuple[
    List[Message],
    Optional[List[Dict]],
    Optional[Dict[str, Any]],
    Optional[Dict],
]:
  """Converts an LlmRequest to litellm inputs and extracts generation params.

  Args:
    llm_request: The LlmRequest to convert.

  Returns:
    The litellm inputs (message list, tool dictionary, response format and
    generation params).
  """
  # 1. Construct messages
  messages: List[Message] = []
  for content in llm_request.contents or []:
    message_param_or_list = _content_to_message_param(content)
    if isinstance(message_param_or_list, list):
      messages.extend(message_param_or_list)
    elif message_param_or_list:  # Ensure it's not None before appending
      messages.append(message_param_or_list)

  if llm_request.config.system_instruction:
    messages.insert(
        0,
        ChatCompletionDeveloperMessage(
            role="developer",
            content=llm_request.config.system_instruction,
        ),
    )

  # 2. Convert tool declarations
  tools: Optional[List[Dict]] = None
  if (
      llm_request.config
      and llm_request.config.tools
      and llm_request.config.tools[0].function_declarations
  ):
    tools = [
        _function_declaration_to_tool_param(tool)
        for tool in llm_request.config.tools[0].function_declarations
    ]

  # 3. Handle response format
  response_format: Optional[Dict[str, Any]] = None
  if llm_request.config and llm_request.config.response_schema:
    response_format = _to_litellm_response_format(
        llm_request.config.response_schema
    )

  # 4. Extract generation parameters
  generation_params: Optional[Dict] = None
  if llm_request.config:
    config_dict = llm_request.config.model_dump(exclude_none=True)
    # Generate LiteLlm parameters here,
    # Following https://docs.litellm.ai/docs/completion/input.
    generation_params = {}
    param_mapping = {
        "max_output_tokens": "max_completion_tokens",
        "stop_sequences": "stop",
    }
    for key in (
        "temperature",
        "max_output_tokens",
        "top_p",
        "top_k",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
    ):
      if key in config_dict:
        mapped_key = param_mapping.get(key, key)
        generation_params[mapped_key] = config_dict[key]

    if not generation_params:
      generation_params = None

  return messages, tools, response_format, generation_params


def _build_function_declaration_log(
    func_decl: types.FunctionDeclaration,
) -> str:
  """Builds a function declaration log.

  Args:
    func_decl: The function declaration to convert.

  Returns:
    The function declaration log.
  """

  param_str = "{}"
  if func_decl.parameters and func_decl.parameters.properties:
    param_str = str({
        k: v.model_dump(exclude_none=True)
        for k, v in func_decl.parameters.properties.items()
    })
  return_str = "None"
  if func_decl.response:
    return_str = str(func_decl.response.model_dump(exclude_none=True))
  return f"{func_decl.name}: {param_str} -> {return_str}"


def _build_request_log(req: LlmRequest) -> str:
  """Builds a request log.

  Args:
    req: The request to convert.

  Returns:
    The request log.
  """

  function_decls: list[types.FunctionDeclaration] = cast(
      list[types.FunctionDeclaration],
      req.config.tools[0].function_declarations if req.config.tools else [],
  )
  function_logs = (
      [
          _build_function_declaration_log(func_decl)
          for func_decl in function_decls
      ]
      if function_decls
      else []
  )
  contents_logs = [
      content.model_dump_json(
          exclude_none=True,
          exclude={
              "parts": {
                  i: _EXCLUDED_PART_FIELD for i in range(len(content.parts))
              }
          },
      )
      for content in req.contents
  ]

  return f"""
LLM Request:
-----------------------------------------------------------
System Instruction:
{req.config.system_instruction}
-----------------------------------------------------------
Contents:
{_NEW_LINE.join(contents_logs)}
-----------------------------------------------------------
Functions:
{_NEW_LINE.join(function_logs)}
-----------------------------------------------------------
"""


def _is_litellm_gemini_model(model_string: str) -> bool:
  """Check if the model is a Gemini model accessed via LiteLLM.

  Args:
    model_string: A LiteLLM model string (e.g., "gemini/gemini-2.5-pro" or
      "vertex_ai/gemini-2.5-flash")

  Returns:
    True if it's a Gemini model accessed via LiteLLM, False otherwise
  """
  # Matches "gemini/gemini-*" (Google AI Studio) or "vertex_ai/gemini-*" (Vertex AI).
  pattern = r"^(gemini|vertex_ai)/gemini-"
  return bool(re.match(pattern, model_string))


def _extract_gemini_model_from_litellm(litellm_model: str) -> str:
  """Extract the pure Gemini model name from a LiteLLM model string.

  Args:
    litellm_model: LiteLLM model string like "gemini/gemini-2.5-pro"

  Returns:
    Pure Gemini model name like "gemini-2.5-pro"
  """
  # Remove LiteLLM provider prefix
  if "/" in litellm_model:
    return litellm_model.split("/", 1)[1]
  return litellm_model


def _warn_gemini_via_litellm(model_string: str) -> None:
  """Warn if Gemini is being used via LiteLLM.

  This function logs a warning suggesting users use Gemini directly rather than
  through LiteLLM for better performance and features.

  Args:
    model_string: The LiteLLM model string to check
  """
  if not _is_litellm_gemini_model(model_string):
    return

  # Check if warning should be suppressed via environment variable
  if os.environ.get(
      "ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", ""
  ).strip().lower() in ("1", "true", "yes", "on"):
    return

  warnings.warn(
      f"[GEMINI_VIA_LITELLM] {model_string}: You are using Gemini via LiteLLM."
      " For better performance, reliability, and access to latest features,"
      " consider using Gemini directly through ADK's native Gemini"
      f" integration. Replace LiteLlm(model='{model_string}') with"
      f" Gemini(model='{_extract_gemini_model_from_litellm(model_string)}')."
      " Set ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS=true to suppress this"
      " warning.",
      category=UserWarning,
      stacklevel=3,
  )


class LiteLlm(BaseLlm):
  """Wrapper around litellm.

  This wrapper can be used with any of the models supported by litellm. The
  environment variable(s) needed for authenticating with the model endpoint must
  be set prior to instantiating this class.

  Example usage:
  ```
  os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
  os.environ["VERTEXAI_LOCATION"] = "your-gcp-location"

  agent = Agent(
      model=LiteLlm(model="vertex_ai/claude-3-7-sonnet@20250219"),
      ...
  )
  ```

  Attributes:
    model: The name of the LiteLlm model.
    llm_client: The LLM client to use for the model.
  """

  llm_client: LiteLLMClient = Field(default_factory=LiteLLMClient)
  """The LLM client to use for the model."""

  _additional_args: Dict[str, Any] = None

  def __init__(self, model: str, **kwargs):
    """Initializes the LiteLlm class.

    Args:
      model: The name of the LiteLlm model.
      **kwargs: Additional arguments to pass to the litellm completion api.
    """
    drop_params = kwargs.pop("drop_params", None)
    super().__init__(model=model, **kwargs)
    # Warn if using Gemini via LiteLLM
    _warn_gemini_via_litellm(model)
    self._additional_args = dict(kwargs)
    # preventing generation call with llm_client
    # and overriding messages, tools and stream which are managed internally
    self._additional_args.pop("llm_client", None)
    self._additional_args.pop("messages", None)
    self._additional_args.pop("tools", None)
    # public api called from runner determines to stream or not
    self._additional_args.pop("stream", None)
    if drop_params is not None:
      self._additional_args["drop_params"] = drop_params

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content asynchronously.

    Args:
      llm_request: LlmRequest, the request to send to the LiteLlm model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """

    self._maybe_append_user_content(llm_request)
    _append_fallback_user_content_if_missing(llm_request)
    logger.debug(_build_request_log(llm_request))

    messages, tools, response_format, generation_params = (
        _get_completion_inputs(llm_request)
    )

    if "functions" in self._additional_args:
      # LiteLLM does not support both tools and functions together.
      tools = None

    completion_args = {
        "model": llm_request.model or self.model,
        "messages": messages,
        "tools": tools,
        "response_format": response_format,
    }
    completion_args.update(self._additional_args)

    if generation_params:
      completion_args.update(generation_params)

    if stream:
      text = ""
      # Track function calls by index
      function_calls = {}  # index -> {name, args, id}
      completion_args["stream"] = True
      completion_args["stream_options"] = {"include_usage": True}
      aggregated_llm_response = None
      aggregated_llm_response_with_tool_call = None
      usage_metadata = None
      fallback_index = 0
      async for part in await self.llm_client.acompletion(**completion_args):
        for chunk, finish_reason in _model_response_to_chunk(part):
          if isinstance(chunk, FunctionChunk):
            index = chunk.index or fallback_index
            if index not in function_calls:
              function_calls[index] = {"name": "", "args": "", "id": None}

            if chunk.name:
              function_calls[index]["name"] += chunk.name
            if chunk.args:
              function_calls[index]["args"] += chunk.args

              # check if args is completed (workaround for improper chunk
              # indexing)
              try:
                json.loads(function_calls[index]["args"])
                fallback_index += 1
              except json.JSONDecodeError:
                pass

            function_calls[index]["id"] = (
                chunk.id or function_calls[index]["id"] or str(index)
            )
          elif isinstance(chunk, TextChunk):
            text += chunk.text
            yield _message_to_generate_content_response(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    content=chunk.text,
                ),
                is_partial=True,
                model_version=part.model,
            )
          elif isinstance(chunk, UsageMetadataChunk):
            usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=chunk.prompt_tokens,
                candidates_token_count=chunk.completion_tokens,
                total_token_count=chunk.total_tokens,
                cached_content_token_count=chunk.cached_prompt_tokens,
            )

          if (
              finish_reason == "tool_calls" or finish_reason == "stop"
          ) and function_calls:
            tool_calls = []
            for index, func_data in function_calls.items():
              if func_data["id"]:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        type="function",
                        id=func_data["id"],
                        function=Function(
                            name=func_data["name"],
                            arguments=func_data["args"],
                            index=index,
                        ),
                    )
                )
            aggregated_llm_response_with_tool_call = (
                _message_to_generate_content_response(
                    ChatCompletionAssistantMessage(
                        role="assistant",
                        content=text,
                        tool_calls=tool_calls,
                    ),
                    model_version=part.model,
                )
            )
            text = ""
            function_calls.clear()
          elif finish_reason == "stop" and text:
            aggregated_llm_response = _message_to_generate_content_response(
                ChatCompletionAssistantMessage(role="assistant", content=text),
                model_version=part.model,
            )
            text = ""

      # waiting until streaming ends to yield the llm_response as litellm tends
      # to send chunk that contains usage_metadata after the chunk with
      # finish_reason set to tool_calls or stop.
      if aggregated_llm_response:
        if usage_metadata:
          aggregated_llm_response.usage_metadata = usage_metadata
          usage_metadata = None
        yield aggregated_llm_response

      if aggregated_llm_response_with_tool_call:
        if usage_metadata:
          aggregated_llm_response_with_tool_call.usage_metadata = usage_metadata
        yield aggregated_llm_response_with_tool_call

    else:
      response = await self.llm_client.acompletion(**completion_args)
      yield _model_response_to_generate_content_response(response)

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    LiteLlm supports all models supported by litellm. We do not keep track of
    these models here. So we return an empty list.

    Returns:
      A list of supported models.
    """

    return []
