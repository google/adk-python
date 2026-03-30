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

"""OCI Generative AI integration for ADK models."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import threading
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

if not TYPE_CHECKING and importlib.util.find_spec("oci") is None:
  raise ImportError(
      "OCI Generative AI support requires: pip install google-adk[oci]"
      "\nOr: pip install oci"
  )

from .base_llm import BaseLlm
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

__all__ = ["OCIGenAILlm"]

logger = logging.getLogger("google_adk." + __name__)


def _to_oci_role(role: Optional[str]) -> str:
  """Map ADK content role to OCI GenAI role string."""
  if role in ("model", "assistant"):
    return "ASSISTANT"
  return "USER"


def _content_to_oci_message(content: types.Content) -> Any:
  """Convert an ADK Content object to an OCI GenAI message.

  OCI GenAI uses:
    - ``UserMessage``      for user turns
    - ``AssistantMessage`` for model turns (may include ``FunctionCall`` items
                           in ``tool_calls``)
    - ``ToolMessage``      for tool results (function_response parts)
  """
  import oci.generative_ai_inference.models as oci_models

  text_parts: list[str] = []
  tool_calls: list[Any] = []
  tool_results: list[tuple[str, str]] = []  # (tool_call_id, result_text)

  for part in content.parts or []:
    if part.text:
      text_parts.append(part.text)
    elif part.function_call:
      # FunctionCall is the OCI subtype of ToolCall that carries name+arguments
      tool_calls.append(
          oci_models.FunctionCall(
              id=part.function_call.id or "",
              type=oci_models.FunctionCall.TYPE_FUNCTION,
              name=part.function_call.name,
              arguments=json.dumps(part.function_call.args or {}),
          )
      )
    elif part.function_response:
      result = part.function_response.response or {}
      tool_results.append((
          part.function_response.id or "",
          json.dumps(result) if isinstance(result, dict) else str(result),
      ))

  role = _to_oci_role(content.role)

  # Tool results map to ToolMessage (one per result)
  if tool_results:
    call_id, result_text = tool_results[0]
    return oci_models.ToolMessage(
        role=oci_models.ToolMessage.ROLE_TOOL,
        tool_call_id=call_id,
        content=[oci_models.TextContent(type="TEXT", text=result_text)],
    )

  if role == "ASSISTANT":
    oci_content = (
        [oci_models.TextContent(type="TEXT", text="\n".join(text_parts))]
        if text_parts
        else []
    )
    return oci_models.AssistantMessage(
        role=oci_models.AssistantMessage.ROLE_ASSISTANT,
        content=oci_content,
        tool_calls=tool_calls or None,
    )

  return oci_models.UserMessage(
      role=oci_models.UserMessage.ROLE_USER,
      content=[
          oci_models.TextContent(type="TEXT", text="\n".join(text_parts))
      ],
  )


def _oci_response_to_llm_response(response: Any) -> LlmResponse:
  """Convert an OCI GenAI chat response to an LlmResponse."""
  chat_response = response.data.chat_response
  parts: list[types.Part] = []
  input_tokens = 0
  output_tokens = 0

  if hasattr(chat_response, "usage"):
    usage = chat_response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

  if hasattr(chat_response, "choices") and chat_response.choices:
    choice = chat_response.choices[0]
    message = getattr(choice, "message", None)
    if message:
      # Text content
      for block in getattr(message, "content", None) or []:
        if hasattr(block, "text") and block.text:
          parts.append(types.Part.from_text(text=block.text))

      # Tool calls — OCI returns FunctionCall objects directly in tool_calls
      for fc in getattr(message, "tool_calls", None) or []:
        args: dict[str, Any] = {}
        try:
          args = json.loads(fc.arguments) if fc.arguments else {}
        except (json.JSONDecodeError, TypeError):
          args = {}
        part = types.Part.from_function_call(
            name=fc.name,
            args=args,
        )
        part.function_call.id = getattr(fc, "id", "") or ""
        parts.append(part)

  return LlmResponse(
      content=types.Content(role="model", parts=parts),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=input_tokens,
          candidates_token_count=output_tokens,
          total_token_count=input_tokens + output_tokens,
      ),
  )


def _function_declaration_to_oci_tool(
    fn: types.FunctionDeclaration,
) -> Any:
  """Convert an ADK FunctionDeclaration to an OCI GenAI Tool."""
  import oci.generative_ai_inference.models as oci_models

  parameters: dict[str, Any] = {"type": "object", "properties": {}}
  if fn.parameters_json_schema:
    parameters = fn.parameters_json_schema
  elif fn.parameters and fn.parameters.properties:
    props = {}
    for k, v in fn.parameters.properties.items():
      props[k] = v.model_dump(by_alias=True, exclude_none=True)
    parameters = {
        "type": "object",
        "properties": props,
    }
    if fn.parameters.required:
      parameters["required"] = fn.parameters.required

  return oci_models.FunctionDefinition(
      type=oci_models.FunctionDefinition.TYPE_FUNCTION,
      name=fn.name,
      description=fn.description or "",
      parameters=parameters,
  )


class OCIGenAILlm(BaseLlm):
  """Integration with OCI Generative AI models.

  Supports models hosted on Oracle Cloud Infrastructure Generative AI service,
  including Meta Llama, Google Gemini, Google Gemma, and other GenericChat
  compatible models.

  Example usage::

      from google.adk.models.oci_genai_llm import OCIGenAILlm
      from google.adk.agents import LlmAgent

      agent = LlmAgent(
          model=OCIGenAILlm(
              model="google.gemini-2.0-flash-001",
              compartment_id="ocid1.compartment.oc1...",
          ),
          ...
      )

  Attributes:
    model: OCI model ID (e.g. ``google.gemini-2.0-flash-001``).
    compartment_id: OCI compartment OCID. Falls back to the
      ``OCI_COMPARTMENT_ID`` environment variable when not set.
    service_endpoint: OCI Generative AI service endpoint URL. Defaults to
      the us-chicago-1 endpoint or ``OCI_SERVICE_ENDPOINT`` env var.
    auth_type: OCI authentication type.  One of ``API_KEY`` (default),
      ``INSTANCE_PRINCIPAL``, or ``RESOURCE_PRINCIPAL``.
    auth_profile: Config profile to use for ``API_KEY`` auth (default:
      ``DEFAULT``).
    auth_file_location: Path to the OCI config file used for ``API_KEY``
      auth (default: ``~/.oci/config``).
    max_tokens: Maximum number of tokens to generate (default: 2048).
  """

  model: str = "google.gemini-2.5-flash"
  compartment_id: Optional[str] = None
  service_endpoint: Optional[str] = None
  auth_type: str = "API_KEY"
  auth_profile: str = "DEFAULT"
  auth_file_location: str = "~/.oci/config"
  max_tokens: int = 2048

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [
        r"meta\.llama-.*",
        r"google\.gemini-.*",
        r"google\.gemma-.*",
        r"xai\.grok-.*",
        r"mistralai\.mistral-.*",
        r"mistralai\.mixtral-.*",
        r"nvidia\..*",
    ]

  @override
  async def generate_content_async(
      self,
      llm_request: LlmRequest,
      stream: bool = False,
  ) -> AsyncGenerator[LlmResponse, None]:
    if stream:
      async for response in self._generate_content_streaming(llm_request):
        yield response
    else:
      response = await asyncio.to_thread(self._call_oci, llm_request)
      yield _oci_response_to_llm_response(response)

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _resolve_compartment_id(self) -> str:
    compartment_id = self.compartment_id or os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
      raise ValueError(
          "compartment_id must be set on OCIGenAILlm or via the"
          " OCI_COMPARTMENT_ID environment variable."
      )
    return compartment_id

  def _resolve_service_endpoint(self) -> str:
    return (
        self.service_endpoint
        or os.environ.get("OCI_SERVICE_ENDPOINT")
        or "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    )

  def _build_client(self, service_endpoint: str) -> Any:
    """Create an OCI GenerativeAiInferenceClient from auth config."""
    import oci
    import oci.auth.signers
    import oci.generative_ai_inference

    if self.auth_type == "INSTANCE_PRINCIPAL":
      signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
      return oci.generative_ai_inference.GenerativeAiInferenceClient(
          config={},
          signer=signer,
          service_endpoint=service_endpoint,
      )
    elif self.auth_type == "RESOURCE_PRINCIPAL":
      signer = oci.auth.signers.get_resource_principals_signer()
      return oci.generative_ai_inference.GenerativeAiInferenceClient(
          config={},
          signer=signer,
          service_endpoint=service_endpoint,
      )
    else:  # API_KEY (default)
      config = oci.config.from_file(
          file_location=self.auth_file_location,
          profile_name=self.auth_profile,
      )
      return oci.generative_ai_inference.GenerativeAiInferenceClient(
          config=config,
          service_endpoint=service_endpoint,
      )

  def _build_chat_details(
      self, llm_request: LlmRequest, is_stream: bool = False
  ) -> Any:
    """Build OCI ChatDetails from an LlmRequest."""
    import oci.generative_ai_inference.models as oci_models

    messages = [
        _content_to_oci_message(c) for c in llm_request.contents or []
    ]

    # Prepend SystemMessage when a system instruction is present
    if llm_request.config and llm_request.config.system_instruction:
      si = llm_request.config.system_instruction
      if isinstance(si, str) and si:
        messages = [
            oci_models.SystemMessage(
                role=oci_models.SystemMessage.ROLE_SYSTEM,
                content=[oci_models.TextContent(type="TEXT", text=si)],
            )
        ] + messages

    # Convert tool declarations if present
    oci_tools: Optional[list[Any]] = None
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
      oci_tools = [
          _function_declaration_to_oci_tool(fn)
          for fn in llm_request.config.tools[0].function_declarations
      ]

    chat_request_kwargs: dict[str, Any] = dict(
        api_format=oci_models.BaseChatRequest.API_FORMAT_GENERIC,
        messages=messages,
        max_tokens=self.max_tokens,
    )
    if oci_tools:
      chat_request_kwargs["tools"] = oci_tools
    if is_stream:
      chat_request_kwargs["is_stream"] = True
      chat_request_kwargs["stream_options"] = oci_models.StreamOptions(
          is_include_usage=True
      )

    return oci_models.ChatDetails(
        compartment_id=self._resolve_compartment_id(),
        serving_mode=oci_models.OnDemandServingMode(model_id=self.model),
        chat_request=oci_models.GenericChatRequest(**chat_request_kwargs),
    )

  def _call_oci(self, llm_request: LlmRequest) -> Any:
    """Synchronous non-streaming OCI GenAI call, run in a thread pool."""
    client = self._build_client(self._resolve_service_endpoint())
    chat_details = self._build_chat_details(llm_request, is_stream=False)
    logger.debug("Sending request to OCI GenAI: model=%s", self.model)
    return client.chat(chat_details)

  def _call_oci_stream(self, llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Synchronous streaming call — collects all SSE event dicts in a thread.

    The OCI SDK wraps an SSE response in ``oci._vendor.sseclient.SSEClient``
    when ``is_stream=True`` is set on the request body.  Each event's
    ``data`` field is an OpenAI-compatible JSON chunk or the sentinel
    ``[DONE]``.
    """
    client = self._build_client(self._resolve_service_endpoint())
    chat_details = self._build_chat_details(llm_request, is_stream=True)
    logger.debug(
        "Sending streaming request to OCI GenAI: model=%s", self.model
    )
    response = client.chat(chat_details)

    chunks: list[dict[str, Any]] = []
    for event in response.data:  # SSEClient is synchronously iterable
      raw = getattr(event, "data", None)
      if not raw or raw.strip() == "[DONE]":
        break
      try:
        chunks.append(json.loads(raw))
      except (json.JSONDecodeError, TypeError):
        logger.debug("Could not parse SSE event data: %r", raw)
    return chunks

  async def _generate_content_streaming(
      self, llm_request: LlmRequest
  ) -> AsyncGenerator[LlmResponse, None]:
    """Yield partial then final LlmResponse from an OCI SSE stream.

    The OCI SDK is fully synchronous, so we collect all SSE chunks in a
    background thread via ``asyncio.to_thread`` and then yield responses
    from the accumulated data — matching the pattern used by
    ``AnthropicLlm._generate_content_streaming``.
    """
    chunks = await asyncio.to_thread(self._call_oci_stream, llm_request)

    text_acc: str = ""
    # tool call accumulators: index → {id, name, arguments}
    tool_acc: dict[int, dict[str, Any]] = {}
    input_tokens: int = 0
    output_tokens: int = 0

    for chunk in chunks:
      # Usage may appear on the final chunk
      if "usage" in chunk and chunk["usage"]:
        u = chunk["usage"]
        input_tokens = u.get("prompt_tokens", 0) or 0
        output_tokens = u.get("completion_tokens", 0) or 0

      for choice in chunk.get("choices", []):
        delta = choice.get("delta") or {}

        # Text delta
        text = delta.get("content") or ""
        if text:
          text_acc += text
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text=text)],
              ),
              partial=True,
          )

        # Tool-call deltas (streamed in pieces across multiple chunks)
        for tc_delta in delta.get("tool_calls") or []:
          idx = tc_delta.get("index", 0)
          if idx not in tool_acc:
            tool_acc[idx] = {"id": "", "name": "", "arguments": ""}
          if tc_delta.get("id"):
            tool_acc[idx]["id"] = tc_delta["id"]
          fn = tc_delta.get("function") or {}
          if fn.get("name"):
            tool_acc[idx]["name"] += fn["name"]
          if fn.get("arguments"):
            tool_acc[idx]["arguments"] += fn["arguments"]

    # Build final aggregated response
    all_parts: list[types.Part] = []
    if text_acc:
      all_parts.append(types.Part.from_text(text=text_acc))
    for tc in sorted(tool_acc.values(), key=lambda x: x.get("name", "")):
      args: dict[str, Any] = {}
      try:
        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
      except (json.JSONDecodeError, TypeError):
        args = {}
      part = types.Part.from_function_call(name=tc["name"], args=args)
      part.function_call.id = tc["id"]
      all_parts.append(part)

    yield LlmResponse(
        content=types.Content(role="model", parts=all_parts),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=input_tokens + output_tokens,
        ),
        partial=False,
    )
