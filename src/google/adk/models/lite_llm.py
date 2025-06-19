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

import asyncio
import base64
import json
import logging
import random
import time
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from google.genai import types
from litellm import acompletion
from litellm import ChatCompletionAssistantMessage
from litellm import ChatCompletionAssistantToolCall
from litellm import ChatCompletionDeveloperMessage
from litellm import ChatCompletionImageUrlObject
from litellm import ChatCompletionMessageToolCall
from litellm import ChatCompletionTextObject
from litellm import ChatCompletionToolMessage
from litellm import ChatCompletionUserMessage
from litellm import ChatCompletionVideoUrlObject
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

logger = logging.getLogger("google_adk." + __name__)

_NEW_LINE = "\n"
_EXCLUDED_PART_FIELD = {"inline_data": {"data"}}

# Retry configuration
RETRIABLE_ERROR_CODES = {429, 500, 502, 503, 504}
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_EXPONENTIAL_BASE = 2.0
DEFAULT_JITTER = True


def _should_retry(exception: Exception) -> bool:
  """Check if an exception should trigger a retry."""
  if hasattr(exception, "status_code"):
    return exception.status_code in RETRIABLE_ERROR_CODES

  # Check for rate limit errors in exception message
  error_msg = str(exception).lower()
  rate_limit_indicators = [
      "rate limit",
      "429",
      "too many requests",
      "quota exceeded",
  ]
  return any(indicator in error_msg for indicator in rate_limit_indicators)


def _calculate_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
  """Calculate delay for exponential backoff with optional jitter."""
  delay = initial_delay * (exponential_base**attempt)
  delay = min(delay, max_delay)

  if jitter:
    # Add random jitter to avoid thundering herd
    delay *= 0.5 + random.random() * 0.5

  return delay


async def _retry_async(
    func,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    jitter: bool = DEFAULT_JITTER,
    **kwargs,
):
  """Async retry wrapper with exponential backoff."""
  last_exception = None

  for attempt in range(max_retries + 1):  # +1 for initial attempt
    try:
      return await func(*args, **kwargs)
    except Exception as e:
      last_exception = e

      if attempt == max_retries or not _should_retry(e):
        logger.error(
            f"Max retries ({max_retries}) exceeded or non-retriable error: {e}"
        )
        raise e

      delay = _calculate_delay(
          attempt, initial_delay, max_delay, exponential_base, jitter
      )
      logger.warning(
          f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
          f"Retrying in {delay:.2f} seconds..."
      )

      await asyncio.sleep(delay)

  raise last_exception


def _retry_sync(
    func,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    jitter: bool = DEFAULT_JITTER,
    **kwargs,
):
  """Sync retry wrapper with exponential backoff."""
  last_exception = None

  for attempt in range(max_retries + 1):  # +1 for initial attempt
    try:
      return func(*args, **kwargs)
    except Exception as e:
      last_exception = e

      if attempt == max_retries or not _should_retry(e):
        logger.error(
            f"Max retries ({max_retries}) exceeded or non-retriable error: {e}"
        )
        raise e

      delay = _calculate_delay(
          attempt, initial_delay, max_delay, exponential_base, jitter
      )
      logger.warning(
          f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
          f"Retrying in {delay:.2f} seconds..."
      )

      time.sleep(delay)

  raise last_exception


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


class LiteLLMClient:
  """Provides acompletion method (for better testability)."""

  def __init__(
      self,
      max_retries: int = DEFAULT_MAX_RETRIES,
      initial_delay: float = DEFAULT_INITIAL_DELAY,
      max_delay: float = DEFAULT_MAX_DELAY,
      exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
      jitter: bool = DEFAULT_JITTER,
  ):
    """Initialize LiteLLMClient with retry configuration.

    Args:
      max_retries: Maximum number of retry attempts (default: 3)
      initial_delay: Initial delay in seconds (default: 1.0)
      max_delay: Maximum delay in seconds (default: 60.0)
      exponential_base: Exponential backoff base (default: 2.0)
      jitter: Whether to add random jitter (default: True)
    """
    self.max_retries = max_retries
    self.initial_delay = initial_delay
    self.max_delay = max_delay
    self.exponential_base = exponential_base
    self.jitter = jitter

  async def acompletion(
      self, model, messages, tools, **kwargs
  ) -> Union[ModelResponse, CustomStreamWrapper]:
    """Asynchronously calls acompletion with retry mechanism.

    Args:
      model: The model name.
      messages: The messages to send to the model.
      tools: The tools to use for the model.
      **kwargs: Additional arguments to pass to acompletion.

    Returns:
      The model response as a message.
    """

    return await _retry_async(
        acompletion,
        model=model,
        messages=messages,
        tools=tools,
        max_retries=self.max_retries,
        initial_delay=self.initial_delay,
        max_delay=self.max_delay,
        exponential_base=self.exponential_base,
        jitter=self.jitter,
        **kwargs,
    )

  def completion(
      self, model, messages, tools, stream=False, **kwargs
  ) -> Union[ModelResponse, CustomStreamWrapper]:
    """Synchronously calls completion with retry mechanism.

    Args:
      model: The model to use.
      messages: The messages to send.
      tools: The tools to use for the model.
      stream: Whether to stream the response.
      **kwargs: Additional arguments to pass to completion.

    Returns:
      The response from the model.
    """

    return _retry_sync(
        completion,
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        max_retries=self.max_retries,
        initial_delay=self.initial_delay,
        max_delay=self.max_delay,
        exponential_base=self.exponential_base,
        jitter=self.jitter,
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
      content_objects.append(
          ChatCompletionTextObject(
              type="text",
              text=part.text,
          )
      )
    elif (
        part.inline_data
        and part.inline_data.data
        and part.inline_data.mime_type
    ):
      base64_string = base64.b64encode(part.inline_data.data).decode("utf-8")
      data_uri = f"data:{part.inline_data.mime_type};base64,{base64_string}"

      if part.inline_data.mime_type.startswith("image"):
        content_objects.append(
            ChatCompletionImageUrlObject(
                type="image_url",
                image_url=data_uri,
            )
        )
      elif part.inline_data.mime_type.startswith("video"):
        content_objects.append(
            ChatCompletionVideoUrlObject(
                type="video_url",
                video_url=data_uri,
            )
        )
      else:
        raise ValueError("LiteLlm(BaseLlm) does not support this content part.")

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
  """Recursively converts a types.Schema to a dictionary.

  Args:
    schema: The schema to convert.

  Returns:
    The dictionary representation of the schema.
  """

  schema_dict = schema.model_dump(exclude_none=True)
  if "type" in schema_dict:
    schema_dict["type"] = schema_dict["type"].lower()
  if "items" in schema_dict:
    if isinstance(schema_dict["items"], dict):
      schema_dict["items"] = _schema_to_dict(
          types.Schema.model_validate(schema_dict["items"])
      )
    elif isinstance(schema_dict["items"]["type"], types.Type):
      schema_dict["items"]["type"] = TYPE_LABELS[
          schema_dict["items"]["type"].value
      ]
  if "properties" in schema_dict:
    properties = {}
    for key, value in schema_dict["properties"].items():
      if isinstance(value, types.Schema):
        properties[key] = _schema_to_dict(value)
      else:
        properties[key] = value
        if "type" in properties[key]:
          properties[key]["type"] = properties[key]["type"].lower()
    schema_dict["properties"] = properties
  return schema_dict


def _function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> dict:
  """Converts a types.FunctionDeclaration to a openapi spec dictionary.

  Args:
    function_declaration: The function declaration to convert.

  Returns:
    The openapi spec dictionary representation of the function declaration.
  """

  assert function_declaration.name

  properties = {}
  if (
      function_declaration.parameters
      and function_declaration.parameters.properties
  ):
    for key, value in function_declaration.parameters.properties.items():
      properties[key] = _schema_to_dict(value)

  return {
      "type": "function",
      "function": {
          "name": function_declaration.name,
          "description": function_declaration.description or "",
          "parameters": {
              "type": "object",
              "properties": properties,
          },
      },
  }


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
          yield FunctionChunk(
              id=tool_call.id,
              name=tool_call.function.name,
              args=tool_call.function.arguments,
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
  if response.get("choices", None):
    message = response["choices"][0].get("message", None)

  if not message:
    raise ValueError("No message in response")

  llm_response = _message_to_generate_content_response(message)
  if response.get("usage", None):
    llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=response["usage"].get("prompt_tokens", 0),
        candidates_token_count=response["usage"].get("completion_tokens", 0),
        total_token_count=response["usage"].get("total_tokens", 0),
    )
  return llm_response


def _message_to_generate_content_response(
    message: Message, is_partial: bool = False
) -> LlmResponse:
  """Converts a litellm message to LlmResponse.

  Args:
    message: The message to convert.
    is_partial: Whether the message is partial.

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
      content=types.Content(role="model", parts=parts), partial=is_partial
  )


def _get_completion_inputs(
    llm_request: LlmRequest,
) -> tuple[Iterable[Message], Iterable[dict]]:
  """Converts an LlmRequest to litellm inputs.

  Args:
    llm_request: The LlmRequest to convert.

  Returns:
    The litellm inputs (message list, tool dictionary and response format).
  """
  messages = []
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

  tools = None
  if (
      llm_request.config
      and llm_request.config.tools
      and llm_request.config.tools[0].function_declarations
  ):
    tools = [
        _function_declaration_to_tool_param(tool)
        for tool in llm_request.config.tools[0].function_declarations
    ]

  response_format = None

  if llm_request.config.response_schema:
    response_format = llm_request.config.response_schema

  return messages, tools, response_format


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


class LiteLlm(BaseLlm):
  """Wrapper around litellm with built-in retry mechanism.

  This wrapper can be used with any of the models supported by litellm. The
  environment variable(s) needed for authenticating with the model endpoint must
  be set prior to instantiating this class.

  Built-in retry mechanism automatically handles transient API errors including:
  - 429 Too Many Requests
  - 500 Internal Server Error
  - 502 Bad Gateway
  - 503 Service Unavailable
  - 504 Gateway Timeout

  Example usage:
  ```
  os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
  os.environ["VERTEXAI_LOCATION"] = "your-gcp-location"

  # Basic usage with default retry settings (3 retries, exponential backoff)
  agent = Agent(
      model=LiteLlm(model="vertex_ai/claude-3-7-sonnet@20250219"),
      ...
  )

  # Custom retry configuration
  agent = Agent(
      model=LiteLlm(
          model="gemini/gemini-2.5-pro-preview-06-05",
          max_retries=5,
          initial_delay=2.0,
          max_delay=120.0,
          api_key=os.getenv("GOOGLE_API_KEY")
      ),
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

  def __init__(
      self,
      model: str,
      max_retries: int = DEFAULT_MAX_RETRIES,
      initial_delay: float = DEFAULT_INITIAL_DELAY,
      max_delay: float = DEFAULT_MAX_DELAY,
      exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
      jitter: bool = DEFAULT_JITTER,
      **kwargs,
  ):
    """Initializes the LiteLlm class.

    Args:
      model: The name of the LiteLlm model.
      max_retries: Maximum number of retry attempts for API errors (default: 3)
      initial_delay: Initial delay in seconds for exponential backoff (default: 1.0)
      max_delay: Maximum delay in seconds for exponential backoff (default: 60.0)
      exponential_base: Exponential backoff base multiplier (default: 2.0)
      jitter: Whether to add random jitter to delays (default: True)
      **kwargs: Additional arguments to pass to the litellm completion api.
    """
    super().__init__(model=model, **kwargs)

    # Initialize retry-aware client
    self.llm_client = LiteLLMClient(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )

    self._additional_args = kwargs
    # preventing generation call with llm_client
    # and overriding messages, tools and stream which are managed internally
    self._additional_args.pop("llm_client", None)
    self._additional_args.pop("messages", None)
    self._additional_args.pop("tools", None)
    # public api called from runner determines to stream or not
    self._additional_args.pop("stream", None)
    # Remove retry parameters from additional args
    self._additional_args.pop("max_retries", None)
    self._additional_args.pop("initial_delay", None)
    self._additional_args.pop("max_delay", None)
    self._additional_args.pop("exponential_base", None)
    self._additional_args.pop("jitter", None)

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
    logger.debug(_build_request_log(llm_request))

    messages, tools, response_format = _get_completion_inputs(llm_request)

    completion_args = {
        "model": self.model,
        "messages": messages,
        "tools": tools,
        "response_format": response_format,
    }
    completion_args.update(self._additional_args)

    if stream:
      text = ""
      # Track function calls by index
      function_calls = {}  # index -> {name, args, id}
      completion_args["stream"] = True
      aggregated_llm_response = None
      aggregated_llm_response_with_tool_call = None
      usage_metadata = None
      fallback_index = 0
      for part in self.llm_client.completion(**completion_args):
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
            )
          elif isinstance(chunk, UsageMetadataChunk):
            usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=chunk.prompt_tokens,
                candidates_token_count=chunk.completion_tokens,
                total_token_count=chunk.total_tokens,
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
                    )
                )
            )
            text = ""
            function_calls.clear()
          elif finish_reason == "stop" and text:
            aggregated_llm_response = _message_to_generate_content_response(
                ChatCompletionAssistantMessage(role="assistant", content=text)
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

  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Provides the list of supported models.

    LiteLlm supports all models supported by litellm. We do not keep track of
    these models here. So we return an empty list.

    Returns:
      A list of supported models.
    """

    return []
