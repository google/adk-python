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
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)

_NEW_LINE = "\n"
_EXCLUDED_PART_FIELD = {"inline_data": {"data"}}


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


# The Portkey SDK is optional.  Import it lazily so that the rest of ADK can
# still be imported even when Portkey isn't installed.  We surface an
# informative error message later if the user actually tries to *use* the
# Portkey-backed LLM.
try:
  from portkey_ai import AsyncPortkey  # type: ignore
  from portkey_ai import Portkey  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
  AsyncPortkey = None  # type: ignore
  Portkey = None  # type: ignore


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
) -> Union[dict, list[dict]]:
  """Converts a types.Content to a Portkey message or list of messages.

  Handles multipart function responses by returning a list of
  tool message objects if multiple function_response parts exist.

  Args:
    content: The content to convert.

  Returns:
    A Portkey message dict or list of message dicts.
  """
  tool_messages = []
  for part in content.parts:
    if part.function_response:
      tool_messages.append({
          "role": "tool",
          "tool_call_id": part.function_response.id,
          "content": _safe_json_serialize(part.function_response.response),
      })
  if tool_messages:
    return tool_messages if len(tool_messages) > 1 else tool_messages[0]

  # Handle user or assistant messages
  role = _to_portkey_role(content.role)
  message_content = _get_content(content.parts) or None

  if role == "user":
    return {"role": "user", "content": message_content}
  else:  # assistant/model
    tool_calls = []
    content_present = False
    for part in content.parts:
      if part.function_call:
        tool_calls.append({
            "type": "function",
            "id": part.function_call.id,
            "function": {
                "name": part.function_call.name,
                "arguments": _safe_json_serialize(part.function_call.args),
            },
        })
      elif part.text or part.inline_data:
        content_present = True

    final_content = message_content if content_present else None
    if final_content and isinstance(final_content, list):
      # when the content is a single text object, we can use it directly
      final_content = (
          final_content[0].get("text", "")
          if final_content[0].get("type", None) == "text"
          else final_content
      )

    message = {"role": role, "content": final_content}
    if tool_calls:
      message["tool_calls"] = tool_calls
    return message


def _get_content(
    parts: Iterable[types.Part],
) -> Union[list[dict], str]:
  """Converts a list of parts to Portkey content.

  Args:
    parts: The parts to convert.

  Returns:
    The Portkey content.
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
      else:
        raise ValueError("Portkey(BaseLlm) does not support this content part.")

  return content_objects


def _to_portkey_role(role: Optional[str]) -> Literal["user", "assistant"]:
  """Converts a types.Content role to a Portkey role.

  Args:
    role: The types.Content role.

  Returns:
    The Portkey role.
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
  """Converts a types.FunctionDeclaration to a tool parameter dictionary.

  Args:
    function_declaration: The function declaration to convert.

  Returns:
    The tool parameter dictionary representation.
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
    response: Any,
) -> Generator[
    Tuple[
        Optional[Union[TextChunk, FunctionChunk, UsageMetadataChunk]],
        Optional[str],
    ],
    None,
    None,
]:
  """Converts a Portkey response to text, function or usage metadata chunk.

  Args:
    response: The response from the model.

  Yields:
    A tuple of text or function or usage metadata chunk and finish reason.
  """
  message = None
  if hasattr(response, 'choices') and response.choices:
    choice = response.choices[0]
    finish_reason = getattr(choice, 'finish_reason', None)
    
    # Check for delta (streaming)
    if hasattr(choice, 'delta') and choice.delta:
      message = choice.delta
    elif hasattr(choice, 'message'):
      message = choice.message

    if message:
      if hasattr(message, 'content') and message.content:
        yield TextChunk(text=message.content), finish_reason

      if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
          if tool_call.type == "function":
            yield FunctionChunk(
                id=tool_call.id,
                name=tool_call.function.name,
                args=tool_call.function.arguments,
                index=getattr(tool_call, 'index', 0),
            ), finish_reason

      if finish_reason and not (
          (hasattr(message, 'content') and message.content) or 
          (hasattr(message, 'tool_calls') and message.tool_calls)
      ):
        yield None, finish_reason

  if not message:
    yield None, None

  # Handle usage metadata
  if hasattr(response, 'usage') and response.usage:
    yield UsageMetadataChunk(
        prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
        completion_tokens=getattr(response.usage, 'completion_tokens', 0),
        total_tokens=getattr(response.usage, 'total_tokens', 0),
    ), None


def _model_response_to_generate_content_response(
    response: Any,
) -> LlmResponse:
  """Converts a Portkey response to LlmResponse. Also adds usage metadata.

  Args:
    response: The model response.

  Returns:
    The LlmResponse.
  """
  message = None
  if hasattr(response, 'choices') and response.choices:
    message = response.choices[0].message

  if not message:
    raise ValueError("No message in response")

  llm_response = _message_to_generate_content_response(message)
  if hasattr(response, 'usage') and response.usage:
    llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=getattr(response.usage, 'prompt_tokens', 0),
        candidates_token_count=getattr(response.usage, 'completion_tokens', 0),
        total_token_count=getattr(response.usage, 'total_tokens', 0),
    )
  return llm_response


def _message_to_generate_content_response(
    message: Any, is_partial: bool = False
) -> LlmResponse:
  """Converts a Portkey message to LlmResponse.

  Args:
    message: The message to convert.
    is_partial: Whether the message is partial.

  Returns:
    The LlmResponse.
  """
  parts = []
  if hasattr(message, 'content') and message.content:
    parts.append(types.Part.from_text(text=message.content))

  if hasattr(message, 'tool_calls') and message.tool_calls:
    for tool_call in message.tool_calls:
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
) -> tuple[Iterable[dict], Iterable[dict], Optional[dict]]:
  """Converts an LlmRequest to Portkey inputs.

  Args:
    llm_request: The LlmRequest to convert.

  Returns:
    The Portkey inputs (message list, tool dictionary and response format).
  """
  messages = []
  for content in llm_request.contents or []:
    message_param_or_list = _content_to_message_param(content)
    if isinstance(message_param_or_list, list):
      messages.extend(message_param_or_list)
    elif message_param_or_list:  # Ensure it's not None before appending
      messages.append(message_param_or_list)

  if llm_request.config and llm_request.config.system_instruction:
    messages.insert(
        0,
        {
            "role": "developer",
            "content": llm_request.config.system_instruction,
        },
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
  if llm_request.config and llm_request.config.response_schema:
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
  if req.config and req.config.tools and req.config.tools[0].function_declarations:
    function_decls: list[types.FunctionDeclaration] = cast(
        list[types.FunctionDeclaration],
        req.config.tools[0].function_declarations,
    )
  else:
    function_decls = []
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
{getattr(req.config, 'system_instruction', '')}
-----------------------------------------------------------
Contents:
{_NEW_LINE.join(contents_logs)}
-----------------------------------------------------------
Functions:
{_NEW_LINE.join(function_logs)}
-----------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Portkey client wrapper (restored)
# ---------------------------------------------------------------------------


class PortkeyClient:
  """Thin wrapper around the Portkey sync/async clients.

  The wrapper exists mainly so we can inject a stub in tests and keep the rest
  of the `PortkeyLlm` implementation dependency-free.
  """

  def __init__(self, **client_kwargs):
    if Portkey is None or AsyncPortkey is None:  # pragma: no cover
      raise ModuleNotFoundError(
          "`portkey_ai` is not installed.  Install it with `pip install "
          "portkey-ai` (or add the extra group 'adk[extensions]') to use "
          "Portkey-backed models."
      )

    self._sync_client = Portkey(**client_kwargs)  # type: ignore[arg-type]
    self._async_client = AsyncPortkey(**client_kwargs)  # type: ignore[arg-type]

  async def acompletion(self, *, messages, tools=None, **kwargs):  # noqa: D401
    """Asynchronous completion helper matching the OpenAI signature."""

    return await self._async_client.chat.completions.create(
        messages=messages,
        tools=tools,
        **kwargs,
    )

  def completion(self, *, messages, tools=None, stream=False, **kwargs):  # noqa: D401
    """Synchronous (potentially streaming) completion helper."""

    return self._sync_client.chat.completions.create(
        messages=messages,
        tools=tools,
        stream=stream,
        **kwargs,
    )


class PortkeyLlm(BaseLlm):
  """Wrapper around Portkey AI Gateway.

  This wrapper can be used with any of the models supported by Portkey's AI Gateway.
  The Portkey API key must be set prior to instantiating this class, along with
  either virtual keys or provider configuration.

  Example usage:
  ```
  os.environ["PORTKEY_API_KEY"] = "your-portkey-api-key"

  # Using virtual key
  agent = Agent(
      model=PortkeyLlm(
          model="gpt-4o",
          virtual_key="your-virtual-key"
      ),
      ...
  )

  # Using provider configuration
  agent = Agent(
      model=PortkeyLlm(
          model="gpt-4o",
          provider="openai",
          Authorization="sk-your-openai-key"
      ),
      ...
  )
  ```

  Attributes:
    model: The name of the model to use.
    portkey_client: The Portkey client to use for the model.
  """

  portkey_client: PortkeyClient = Field(default=None)
  """The Portkey client to use for the model."""

  _additional_args: Dict[str, Any] = None

  def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
    """Initializes the PortkeyLlm class.

    Args:
      model: The name of the model to use.
      api_key: The Portkey API key (defaults to PORTKEY_API_KEY env var).
      **kwargs: Additional arguments to pass to the Portkey client and completion API.
    """
    super().__init__(model=model, **kwargs)
    
    # Extract Portkey-specific client arguments
    client_args = {}
    if api_key:
      client_args["api_key"] = api_key
    
    # Handle virtual_key or config
    if "virtual_key" in kwargs:
      client_args["virtual_key"] = kwargs.pop("virtual_key")
    elif "config" in kwargs:
      client_args["config"] = kwargs.pop("config")
    elif "provider" in kwargs:
      client_args["provider"] = kwargs.pop("provider")
      # Handle provider-specific auth headers
      if "Authorization" in kwargs:
        client_args["Authorization"] = kwargs.pop("Authorization")
    
    if Portkey is None or AsyncPortkey is None:  # pragma: no cover
      raise ModuleNotFoundError(
          "`portkey_ai` is not installed.  Install it with `pip install "
          "portkey-ai` (or add the extra group 'adk[extensions]' if that "
          "covers it) to use Portkey-backed models."
      )

    self.portkey_client = PortkeyClient(**client_args)
    
    # Store remaining args for completion calls
    self._additional_args = kwargs
    # Remove args that are managed internally
    self._additional_args.pop("messages", None)
    self._additional_args.pop("tools", None)
    self._additional_args.pop("stream", None)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content asynchronously.

    Args:
      llm_request: LlmRequest, the request to send to the Portkey model.
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
      
      for part in self.portkey_client.completion(**completion_args):
        for chunk, finish_reason in _model_response_to_chunk(part):
          if isinstance(chunk, FunctionChunk):
            index = chunk.index or fallback_index
            if index not in function_calls:
              function_calls[index] = {"name": "", "args": "", "id": None}

            if chunk.name:
              function_calls[index]["name"] += chunk.name
            if chunk.args:
              function_calls[index]["args"] += chunk.args

              # check if args is completed (workaround for improper chunk indexing)
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
                type('Message', (), {
                    'content': chunk.text,
                    'tool_calls': None
                })(),
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
                    type('ToolCall', (), {
                        'type': 'function',
                        'id': func_data["id"],
                        'function': type('Function', (), {
                            'name': func_data["name"],
                            'arguments': func_data["args"],
                        })()
                    })()
                )
            aggregated_llm_response_with_tool_call = (
                _message_to_generate_content_response(
                    type('Message', (), {
                        'content': "",
                        'tool_calls': tool_calls
                    })()
                )
            )
            function_calls.clear()
          elif finish_reason == "stop" and text:
            aggregated_llm_response = _message_to_generate_content_response(
                type('Message', (), {
                    'content': text,
                    'tool_calls': None
                })()
            )
            text = ""

      # waiting until streaming ends to yield the llm_response
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
      response = await self.portkey_client.acompletion(**completion_args)
      yield _model_response_to_generate_content_response(response)

  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Provides the list of supported models.

    Portkey supports 200+ models across 30+ providers through its AI Gateway.
    We do not keep track of these models here. So we return an empty list.

    Returns:
      A list of supported models.
    """
    return []