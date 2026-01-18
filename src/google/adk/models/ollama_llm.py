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
import json
import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import Sequence
from typing import Union
import urllib.error
import urllib.request

from google.genai import types
from pydantic import Field
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)

_CHAT_ENDPOINT = "/api/chat"


class Ollama(BaseLlm):
  """Native integration for Ollama-hosted models.

  This backend talks directly to the Ollama HTTP API:

      POST /api/chat

  It supports:
    * `ollama/<model>` names (e.g. `ollama/llama3.2`)
    * `ollama_chat/<model>` names for LiteLlm compatibility
    * System / user / assistant messages
    * Unary generation
    * Tool-calling via Ollama `tools` schema
  """

  # Default model name is compatible with Agent(model="ollama/llama3.1")
  model: str = "ollama/llama3.1"

  host: str = Field(
      default=os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"),
      description="Base URL of the Ollama server.",
  )
  request_timeout: float = Field(
      default=120.0,
      description="Timeout in seconds for Ollama requests.",
  )

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    # Allow any `ollama/...` style name.
    return [r"ollama\/.+"]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    if stream:
      logger.warning(
          "Streaming is not yet supported for Ollama; falling back to unary."
      )

    # Ensure last user content is appended if needed (BaseLlm helper).
    self._maybe_append_user_content(llm_request)

    payload = self._build_payload(llm_request)
    try:
      response_json = await asyncio.to_thread(self._post_chat, payload)
    except RuntimeError as exc:
      logger.error("Failed to call Ollama: %s", exc)
      yield LlmResponse(error_code="OLLAMA_ERROR", error_message=str(exc))
      return

    llm_response = self._to_llm_response(
        response_json, request_model=llm_request.model
    )
    yield llm_response

  # ---------------------------------------------------------------------------
  # Payload construction
  # ---------------------------------------------------------------------------

  def _build_payload(self, llm_request: LlmRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": self._extract_model_name(llm_request.model),
        "messages": self._convert_messages(llm_request),
        "stream": False,
    }

    if tools := self._convert_tools(llm_request):
      payload["tools"] = tools
    if options := self._convert_options(llm_request):
      payload["options"] = options

    return payload

  def _extract_model_name(self, request_model: Optional[str]) -> str:
    """Normalize model name for the Ollama API.

    Supports:
      * "ollama/llama3.2"      → "llama3.2"
      * "ollama_chat/llama3.2" → "llama3.2"
      * "llama3.2"             → "llama3.2"
    """
    model_name = request_model or self.model
    if model_name.startswith("ollama/") or model_name.startswith(
        "ollama_chat/"
    ):
      return model_name.split("/", 1)[1]
    return model_name

  def _convert_messages(self, llm_request: LlmRequest) -> list[dict[str, str]]:
    """Convert ADK Contents into Ollama chat messages."""
    messages: list[dict[str, str]] = []

    # System instruction → first system message.
    system_instruction = llm_request.config.system_instruction
    if system_instruction:
      messages.append({
          "role": "system",
          "content": self._system_instruction_to_text(system_instruction),
      })

    # User / assistant / tool messages.
    for content in llm_request.contents:
      message_text = self._content_to_text(content)
      if not message_text:
        continue
      role = self._map_role(content.role)
      messages.append({"role": role, "content": message_text})

    return messages

  def _system_instruction_to_text(self, system_instruction: Any) -> str:
    """Normalize `system_instruction` into plain text.

    It may be:
      * a plain string
      * a types.Content object
      * a list/tuple of types.Content and/or strings
    """
    # Single Content object
    if isinstance(system_instruction, types.Content):
      return self._content_to_text(system_instruction)

    # Sequence of items (e.g. list[Content])
    if isinstance(system_instruction, (list, tuple)):
      pieces: list[str] = []
      for item in system_instruction:
        if isinstance(item, types.Content):
          pieces.append(self._content_to_text(item))
        elif item is not None:
          pieces.append(str(item))
      return "\n".join(pieces)

    # Fallback: assume it's already string-like
    return str(system_instruction)

  def _content_to_text(self, content: types.Content) -> str:
    """Flatten a `Content` into plain text for Ollama.

    Encodes tool calls and tool responses as tagged lines so that the model
    can reason about them and generate new tool calls.
    """
    parts = content.parts or []
    text_parts: list[str] = []

    for part in parts:
      if part.text:
        text_parts.append(part.text)

      elif part.function_response:
        # Tool result from a previous call.
        try:
          response_json = json.dumps(
              part.function_response.response, ensure_ascii=False
          )
        except TypeError:
          response_json = str(part.function_response.response)
        text_parts.append(
            f"[tool_response name={part.function_response.name or ''}]"
            f" {response_json}"
        )

      elif part.function_call:
        # A model-issued tool call (arguments as JSON).
        try:
          args_json = json.dumps(part.function_call.args, ensure_ascii=False)
        except TypeError:
          args_json = str(part.function_call.args)
        text_parts.append(
            f"[tool_call name={part.function_call.name}] {args_json}"
        )

      else:
        logger.debug(
            "Skipping unsupported content part for Ollama message: %s", part
        )

    return "\n".join(text_parts)

  def _map_role(self, role: Optional[str]) -> str:
    if role in ("model", "assistant"):
      return "assistant"
    if role == "system":
      return "system"
    # "user", "tool", or anything else defaults to "user".
    return "user"

  def _convert_tools(self, llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Convert ADK tool declarations into Ollama tool schema."""
    tools_spec: list[dict[str, Any]] = []
    if not llm_request.config.tools:
      return tools_spec

    for tool in llm_request.config.tools:
      function_declarations: Optional[Sequence[types.FunctionDeclaration]] = (
          tool.function_declarations if isinstance(tool, types.Tool) else None
      )
      if not function_declarations:
        continue

      for function_declaration in function_declarations:
        tools_spec.append({
            "type": "function",
            "function": {
                "name": function_declaration.name,
                "description": function_declaration.description or "",
                "parameters": self._function_parameters_to_json(
                    function_declaration
                ),
            },
        })

    return tools_spec

  def _function_parameters_to_json(
      self, function_declaration: types.FunctionDeclaration
  ) -> dict[str, Any]:
    """Convert function parameters Schema → JSON Schema for Ollama."""
    if function_declaration.parameters is None:
      return {"type": "object", "properties": {}}

    try:
      return function_declaration.parameters.model_dump(exclude_none=True)
    except AttributeError:
      # model_dump is not guaranteed depending on the genai version.
      try:
        return json.loads(
            function_declaration.parameters.model_dump_json(exclude_none=True)
        )
      except (AttributeError, json.JSONDecodeError, TypeError) as exc:
        logger.debug(
            "Failed to convert function parameters, defaulting to empty"
            " schema: %s",
            exc,
        )
        return {"type": "object", "properties": {}}

  def _convert_options(self, llm_request: LlmRequest) -> dict[str, Any]:
    """Map ADK generation config fields to Ollama options."""
    options: dict[str, Any] = {}
    config = llm_request.config

    temperature = getattr(config, "temperature", None)
    if temperature is not None:
      options["temperature"] = temperature

    top_p = getattr(config, "top_p", None)
    if top_p is not None:
      options["top_p"] = top_p

    max_output_tokens = getattr(config, "max_output_tokens", None)
    if max_output_tokens is not None:
      # Ollama uses `num_predict` to limit generated tokens.
      options["num_predict"] = max_output_tokens

    return options

  # ---------------------------------------------------------------------------
  # HTTP call
  # ---------------------------------------------------------------------------

  def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
    """Perform a blocking POST /api/chat call to Ollama.
    Note: This method is intentionally blocking and is executed via
    asyncio.to_thread() to avoid introducing additional async HTTP
    dependencies. This keeps the backend consistent with existing ADK
    providers.
    """

    url = self.host.rstrip("/") + _CHAT_ENDPOINT
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
      with urllib.request.urlopen(
          request, timeout=self.request_timeout
      ) as response:
        response_body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
      raise RuntimeError(exc.reason) from exc
    except urllib.error.HTTPError as exc:
      message = exc.read().decode("utf-8", errors="ignore")
      raise RuntimeError(f"{exc.code}: {message}") from exc

    return json.loads(response_body)

  # ---------------------------------------------------------------------------
  # Response mapping
  # ---------------------------------------------------------------------------

  def _to_llm_response(
      self,
      response_json: dict[str, Any],
      request_model: Optional[str] = None,
  ) -> LlmResponse:
    """Convert Ollama JSON response → ADK `LlmResponse`."""
    if error := response_json.get("error"):
      return LlmResponse(
          error_code="OLLAMA_ERROR",
          error_message=str(error),
      )

    message = response_json.get("message", {}) or {}
    parts: list[types.Part] = []

    # 1) Main text content.
    content = message.get("content")
    if isinstance(content, str) and content.strip():
      parts.append(types.Part.from_text(text=content))

    # 2) Tool calls (if any).
    for tool_call in message.get("tool_calls", []):
      function_payload = tool_call.get("function", {}) or {}
      name = function_payload.get("name")
      if not name:
        logger.warning("Skipping tool call with missing name: %s", tool_call)
        continue
      arguments: Union[str, dict[str, Any], None] = function_payload.get(
          "arguments"
      )

      if isinstance(arguments, str):
        try:
          arguments = json.loads(arguments)
        except json.JSONDecodeError:
          logger.warning(
              "Failed to parse tool call arguments as JSON: %s. Defaulting to"
              " empty arguments.",
              arguments,
          )
          arguments = {}
      elif arguments is None:
        arguments = {}

      function_call = types.FunctionCall(name=name, args=arguments)
      if tool_call_id := tool_call.get("id"):
        # id is useful for correlating tool_call ↔ tool_response.
        setattr(function_call, "id", tool_call_id)

      parts.append(types.Part(function_call=function_call))

    if not parts:
      return LlmResponse(
          error_code="NO_CONTENT",
          error_message="Ollama response did not contain model output.",
      )

    # 3) Usage mapping (Ollama → GenerateContentResponseUsageMetadata).
    # Ollama returns:
    #   prompt_eval_count: tokens in prompt
    #   eval_count:        tokens in completion
    prompt_tokens = response_json.get("prompt_eval_count")
    completion_tokens = response_json.get("eval_count")

    # Fallback: if someone wraps usage in a dict (e.g. in tests).
    if prompt_tokens is None or completion_tokens is None:
      usage = response_json.get("usage") or {}
      if prompt_tokens is None:
        prompt_tokens = usage.get("prompt_tokens")
      if completion_tokens is None:
        completion_tokens = usage.get("completion_tokens")

    usage_metadata: Optional[types.GenerateContentResponseUsageMetadata] = None
    if prompt_tokens is not None and completion_tokens is not None:
      total_tokens = response_json.get("total_tokens")
      if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
      usage_metadata = types.GenerateContentResponseUsageMetadata(
          prompt_token_count=prompt_tokens,
          candidates_token_count=completion_tokens,
          total_token_count=total_tokens,
      )

    # 4) Model version: prefer Ollama's `model`, fallback to request model.
    model_version = response_json.get("model") or self._extract_model_name(
        request_model or self.model
    )

    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        model_version=model_version,
        usage_metadata=usage_metadata,
    )
