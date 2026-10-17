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

from collections.abc import Awaitable
from collections.abc import Callable
import copy
from functools import cached_property
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Literal

from google.genai import types

try:
  from openai import AsyncOpenAI
  from openai.types import CompletionUsage
  from openai.types.chat import ChatCompletion
  from openai.types.chat import ChatCompletionChunk  # noqa: F401
  from openai.types.chat import ChatCompletionContentPartImageParam
  from openai.types.chat import ChatCompletionMessage
  from openai.types.chat import ChatCompletionMessageParam
  from openai.types.chat import ChatCompletionToolParam
except ImportError as e:
  raise ImportError(
      "The 'openai' package is not installed. Please install it with "
      '`pip install "google-adk[openai]"` to use the OpenAILlm.'
  ) from e

from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from . import _openai_common
from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...utils import streaming_utils
from ...utils._schema_utils import lowercase_schema_types
from ._openai_schema import enforce_strict_openai_schema

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


_serialize_system_instruction = _openai_common.serialize_system_instruction
_tool_choice = _openai_common.tool_choice
# The finish-reason mapper lives in _openai_common; alias it under the private
# name this module and its tests use.
_map_finish_reason = _openai_common.map_finish_reason
_is_reasoning_model = _openai_common.is_reasoning_model


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


def _function_declaration_to_openai_tool(
    function_declaration: types.FunctionDeclaration,
) -> ChatCompletionToolParam:
  """Converts a function declaration to an OpenAI tool param."""
  if not function_declaration.name:
    raise ValueError("FunctionDeclaration must have a name.")

  # Use parameters_json_schema if available, otherwise convert from parameters
  if function_declaration.parameters_json_schema:
    parameters = copy.deepcopy(function_declaration.parameters_json_schema)
    lowercase_schema_types(parameters)
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
    lowercase_schema_types(parameters)

  return {
      "type": "function",
      "function": {
          "name": function_declaration.name,
          "description": function_declaration.description or "",
          "parameters": parameters,
      },
  }


def _extract_cached_token_count(usage: CompletionUsage) -> int | None:
  """Returns OpenAI prompt_tokens_details.cached_tokens, if present."""
  details = getattr(usage, "prompt_tokens_details", None)
  cached = getattr(details, "cached_tokens", None)
  return cached if isinstance(cached, int) else None


def _extract_reasoning_token_count(usage: CompletionUsage) -> int | None:
  """Returns OpenAI completion_tokens_details.reasoning_tokens, if present."""
  details = getattr(usage, "completion_tokens_details", None)
  reasoning = getattr(details, "reasoning_tokens", None)
  return reasoning if isinstance(reasoning, int) else None


def _usage_metadata(
    usage: CompletionUsage | None,
) -> types.GenerateContentResponseUsageMetadata | None:
  """Builds ADK usage metadata, tolerating endpoints that omit usage."""
  if usage is None:
    return None
  return types.GenerateContentResponseUsageMetadata(
      prompt_token_count=usage.prompt_tokens,
      candidates_token_count=usage.completion_tokens,
      total_token_count=usage.total_tokens,
      cached_content_token_count=_extract_cached_token_count(usage),
      # Reasoning tokens are also counted in completion_tokens, matching the
      # Responses surface's candidates/thoughts mapping. Unlike Gemini, where
      # the two buckets are disjoint, thoughts here is a subset of candidates,
      # so telemetry/_token_usage.py (which sums candidates + thoughts into
      # output tokens) over-counts reasoning tokens for OpenAI models. That
      # aggregator needs to learn about overlapping buckets; until then this
      # keeps both OpenAI surfaces consistent.
      thoughts_token_count=_extract_reasoning_token_count(usage),
  )


def _tool_call_parts(message: ChatCompletionMessage) -> list[types.Part]:
  """Converts OpenAI tool calls on a message to ADK function-call parts."""
  parts: list[types.Part] = []
  for tool_call in message.tool_calls or []:
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
  return parts


def _response_to_llm_response(response: ChatCompletion) -> LlmResponse:
  """Parses an OpenAI response into an LlmResponse."""
  usage = getattr(response, "usage", None)
  if not response.choices:
    # OpenAI-compatible backends occasionally return no choices (e.g. when a
    # request is filtered). Surface it as an error rather than raising.
    return LlmResponse(
        error_code=types.FinishReason.OTHER,
        error_message="OpenAI response contained no choices.",
        finish_reason=types.FinishReason.OTHER,
        usage_metadata=_usage_metadata(usage),
    )

  choice = response.choices[0]
  message = choice.message

  parts = []
  if message.content:
    parts.append(types.Part.from_text(text=message.content))
  parts.extend(_tool_call_parts(message))

  raw_finish_reason = getattr(choice, "finish_reason", None)
  finish_reason = _map_finish_reason(raw_finish_reason)

  if not parts and finish_reason not in (None, types.FinishReason.STOP):
    # No usable content and the model stopped for an abnormal reason (e.g.
    # content filtering or hitting the token limit before emitting anything).
    # Mirror LlmResponse.create and surface it as an error. A truncated-but-
    # usable response (content present with a non-STOP reason) stays a success.
    return LlmResponse(
        error_code=finish_reason,
        error_message=(
            f"OpenAI response finished with reason {raw_finish_reason!r} and"
            " no content."
        ),
        finish_reason=finish_reason,
        usage_metadata=_usage_metadata(usage),
    )

  return LlmResponse(
      content=types.Content(role="model", parts=parts) if parts else None,
      usage_metadata=_usage_metadata(usage),
      finish_reason=finish_reason,
  )


class OpenAILlm(BaseLlm):
  """Integration with OpenAI models.

  Set ``api_key`` and ``base_url`` to reach the default OpenAI host or any
  OpenAI-compatible backend (for example xAI Grok on Vertex AI, whose
  ``base_url`` is the ``endpoints/openapi`` surface and whose ``api_key`` is a
  Google Cloud access token). ``api_key`` may be a string or a zero-arg callable
  (sync or async) that returns one, so a rotating credential can be plugged in.
  For anything the client supports beyond these (organization, timeout, retries,
  custom headers, ...), pass a pre-configured ``AsyncOpenAI`` instance as
  ``client``.

  Attributes:
      model: The name of the OpenAI model.
      max_tokens: The maximum number of tokens to generate. For reasoning models
        this is sent as ``max_completion_tokens``, which also covers hidden
        reasoning tokens; a budget too small for the reasoning phase yields an
        empty response with ``finish_reason`` ``length`` (surfaced as a
        MAX_TOKENS error, not silently), so raise this for reasoning models that
        need visible output.
      api_key: The API key, either as a string or as a zero-argument callable
        returning a string (or an awaitable of one). ``AsyncOpenAI`` re-invokes
        a callable on every request, so it can supply a credential that expires
        and must be refreshed (e.g. a Vertex AI OAuth bearer token, which lives
        ~1h). Ignored when ``client`` is set.
      base_url: Base URL of the OpenAI-compatible host. Ignored when ``client``
        is set.
      client: A pre-configured OpenAI client. When unset, a default client is
        constructed from ``api_key``/``base_url`` and the environment.
  """

  model: str = "gpt-4o"
  max_tokens: int = 4096
  api_key: str | Callable[[], str] | Callable[[], Awaitable[str]] | None = (
      Field(default=None, exclude=True, repr=False)
  )
  base_url: str | None = None
  client: AsyncOpenAI | None = None

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r"gpt-.*", r"o\d+-.*"]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    messages: list[Any] = []
    if llm_request.config and llm_request.config.system_instruction:
      system_text = _serialize_system_instruction(
          llm_request.config.system_instruction
      )
      if system_text:
        messages.append({"role": "system", "content": system_text})

    for content in llm_request.contents or []:
      messages.extend(_content_to_openai_messages(content))

    tools = []
    if llm_request.config and llm_request.config.tools:
      for tool in llm_request.config.tools:
        if not tool.function_declarations:
          logger.warning(
              "Skipping a tool with no function declarations; only function"
              " tools are supported on the Chat Completions API."
          )
          continue
        for function_declaration in tool.function_declarations:
          tools.append(
              _function_declaration_to_openai_tool(function_declaration)
          )

    tool_choice = None
    if tools:
      tool_choice = _tool_choice(llm_request.config) or "auto"

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
        enforce_strict_openai_schema(schema_dict)
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

    kwargs: dict[str, Any] = {
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

    # Reasoning models (o-series, gpt-5.x, gpt-6.x) reject ``max_tokens`` (they
    # require ``max_completion_tokens``) and reject a non-default
    # ``temperature`` / ``top_p``. Normalize the assembled kwargs once so both
    # the ``self.max_tokens`` and ``config.max_output_tokens`` write sites, and
    # the streaming and non-streaming paths, are covered.
    if _is_reasoning_model(self.model):
      # ``max_tokens`` is always present (self.max_tokens is a non-optional int
      # and any config override is also non-None), so move it unconditionally.
      kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    # Reasoning models reject a non-default temperature/top_p; strip either from
    # the request (with a warning) when it would 400.
    _openai_common.strip_unsupported_sampling_params(kwargs, self.model)

    # Reasoning effort (from OpenAIGenerateContentConfig.effort) maps to the
    # flat ``reasoning_effort`` Chat Completions parameter. The tier is
    # validated against the model only when the request targets the real
    # OpenAI backend (see ``targets_default_openai_host``); otherwise it is
    # passed through for the compatible backend to accept or reject.
    validate = _openai_common.targets_default_openai_host(
        client=self.client,
        base_url=self.base_url,
    )
    effort = _openai_common.build_reasoning_effort(
        llm_request.config,
        self.model,
        "chat",
        validate=validate,
    )
    if effort is not None:
      kwargs["reasoning_effort"] = effort

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
    # Ask for the trailing usage-only chunk. Backends that ignore this simply
    # never send it, so the metadata stays absent rather than the call failing.
    kwargs["stream_options"] = {"include_usage": True}
    raw_stream = await self._openai_client.chat.completions.create(**kwargs)

    text_accumulated = ""
    tool_calls_accumulated: dict[int, dict[str, Any]] = {}
    usage: CompletionUsage | None = None
    finish_reason: str | None = None

    async for chunk in raw_stream:
      if getattr(chunk, "usage", None):
        usage = chunk.usage
      if not chunk.choices:
        continue
      choice = chunk.choices[0]
      if getattr(choice, "finish_reason", None):
        finish_reason = choice.finish_reason
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
                "name": tc_delta.function.name if tc_delta.function else None,
                "arguments": "",
            }
          else:
            if tc_delta.id:
              tool_calls_accumulated[index]["id"] = tc_delta.id
            if tc_delta.function and tc_delta.function.name:
              tool_calls_accumulated[index]["name"] = tc_delta.function.name

          arguments_delta = (
              tc_delta.function.arguments if tc_delta.function else None
          )
          partial_args = None
          if arguments_delta:
            tool_calls_accumulated[index]["arguments"] += arguments_delta
            tracker = tool_calls_accumulated[index].setdefault(
                "tracker", streaming_utils._JsonPathTracker()
            )
            partial_args = tracker.handle_chunk(arguments_delta)

          yield LlmResponse(
              partial=True,
              content=types.Content(
                  role="model",
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id=tool_calls_accumulated[index]["id"],
                              name=tool_calls_accumulated[index]["name"]
                              or None,
                              partial_args=partial_args or None,
                              will_continue=True,
                          )
                      )
                  ],
              ),
          )

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

    mapped_finish_reason = _map_finish_reason(finish_reason)

    if not parts and mapped_finish_reason not in (
        None,
        types.FinishReason.STOP,
    ):
      # Nothing was streamed and the model stopped for an abnormal reason (e.g.
      # content filtering, or hitting the token limit before emitting anything).
      # Mirror the non-streaming path and surface it as an error rather than a
      # silent empty final chunk.
      yield LlmResponse(
          error_code=mapped_finish_reason,
          error_message=(
              "OpenAI streaming response finished with reason"
              f" {finish_reason!r} and no content."
          ),
          finish_reason=mapped_finish_reason,
          usage_metadata=_usage_metadata(usage),
      )
      return

    # Final chunk. Unlike the non-streaming path (which returns content=None for
    # an empty body), the streaming path always emits a non-partial closing
    # response so the trailing usage-only chunk still delivers usage_metadata to
    # the caller; a None content here would be dropped downstream and the token
    # usage lost. An empty parts list is therefore expected and intentional.
    yield LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=False,
        usage_metadata=_usage_metadata(usage),
        finish_reason=mapped_finish_reason,
    )

  @cached_property
  def _openai_client(self) -> AsyncOpenAI:
    if self.client is not None:
      return self.client
    kwargs: dict[str, Any] = {}
    api_key = _openai_common.build_api_key(self.api_key)
    if api_key is not None:
      kwargs["api_key"] = api_key
    if self.base_url is not None:
      kwargs["base_url"] = self.base_url
    # ``AsyncOpenAI`` awaits a callable api_key on every request, so an
    # expiring credential is refreshed without rebuilding the client.
    return AsyncOpenAI(**kwargs)
