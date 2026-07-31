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
import base64
from functools import cached_property
import importlib.util
import json
import logging
import os
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

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse

if TYPE_CHECKING:
  from google.adk.models.llm_request import LlmRequest

__all__ = ["OCIGenAILlm"]

logger = logging.getLogger("google_adk." + __name__)


def _to_oci_role(role: Optional[str]) -> str:
  """Map ADK content role to OCI GenAI role string."""
  if role in ("model", "assistant"):
    return "ASSISTANT"
  return "USER"


def _build_response_format(
    cfg: types.GenerateContentConfig, oci_models: Any
) -> Optional[Any]:
  """Map google.genai response config to OCI ResponseFormat.

  - ``response_schema`` (Pydantic class, dict, or genai Schema) →
    ``JsonSchemaResponseFormat`` (strict structured output).
  - ``response_mime_type == "application/json"`` only →
    ``JsonObjectResponseFormat``.
  - ``response_mime_type == "text/plain"`` → ``TextResponseFormat`` (default
    behaviour; only emitted when explicitly requested).
  """
  schema = cfg.response_schema
  mime = cfg.response_mime_type or ""

  if schema is not None:
    schema_dict: dict[str, Any]
    if hasattr(schema, "model_json_schema"):
      # Pydantic v2 model class
      schema_dict = schema.model_json_schema()
    elif hasattr(schema, "to_json_dict"):
      # google.genai Schema instance
      schema_dict = schema.to_json_dict()
    elif isinstance(schema, dict):
      schema_dict = schema
    else:
      return None
    return oci_models.JsonSchemaResponseFormat(
        type="JSON_SCHEMA",
        json_schema=oci_models.ResponseJsonSchema(
            name=schema_dict.get("title", "response"),
            description=schema_dict.get("description"),
            schema=schema_dict,
            is_strict=True,
        ),
    )

  if mime == "application/json":
    return oci_models.JsonObjectResponseFormat(type="JSON_OBJECT")
  if mime == "text/plain":
    return oci_models.TextResponseFormat(type="TEXT")
  return None


def _media_blocks_for_part(part: types.Part) -> list[Any]:
  """Map a multimodal Part (inline_data / file_data) to OCI ChatContent blocks.

  OCI Generative AI Inference (/20231130/) accepts ImageContent / AudioContent /
  VideoContent / DocumentContent, each carrying a URL in ``{kind}_url.url``.
  Inline bytes are wrapped as ``data:<mime>;base64,<...>``; file_data passes
  the ``file_uri`` through.

  Returns an empty list for parts that have no media payload.
  """
  import oci.generative_ai_inference.models as oci_models

  url: Optional[str] = None
  mime: Optional[str] = None

  if part.inline_data and part.inline_data.data is not None:
    mime = part.inline_data.mime_type or "application/octet-stream"
    raw = part.inline_data.data
    if isinstance(raw, (bytes, bytearray)):
      encoded = base64.b64encode(bytes(raw)).decode("ascii")
    else:
      encoded = str(raw)
    url = f"data:{mime};base64,{encoded}"
  elif part.file_data and part.file_data.file_uri:
    url = part.file_data.file_uri
    mime = part.file_data.mime_type

  if not url:
    return []

  category = (mime or "").split("/", 1)[0].lower()
  if category == "image":
    return [
        oci_models.ImageContent(
            type="IMAGE", image_url=oci_models.ImageUrl(url=url)
        )
    ]
  if category == "audio":
    return [
        oci_models.AudioContent(
            type="AUDIO", audio_url=oci_models.AudioUrl(url=url)
        )
    ]
  if category == "video":
    return [
        oci_models.VideoContent(
            type="VIDEO", video_url=oci_models.VideoUrl(url=url)
        )
    ]
  # Documents (application/pdf, text/*, etc.) and any other mime
  return [
      oci_models.DocumentContent(
          type="DOCUMENT", document_url=oci_models.DocumentUrl(url=url)
      )
  ]


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
  media_blocks: list[Any] = []
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
    elif part.inline_data or part.file_data:
      media_blocks.extend(_media_blocks_for_part(part))

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
    oci_content: list[Any] = []
    if text_parts:
      oci_content.append(
          oci_models.TextContent(type="TEXT", text="\n".join(text_parts))
      )
    return oci_models.AssistantMessage(
        role=oci_models.AssistantMessage.ROLE_ASSISTANT,
        content=oci_content,
        tool_calls=tool_calls or None,
    )

  user_content: list[Any] = []
  if text_parts:
    user_content.append(
        oci_models.TextContent(type="TEXT", text="\n".join(text_parts))
    )
  user_content.extend(media_blocks)
  return oci_models.UserMessage(
      role=oci_models.UserMessage.ROLE_USER,
      content=user_content,
  )


def _oci_response_to_llm_response(response: Any) -> LlmResponse:
  """Convert an OCI GenAI chat response to an LlmResponse."""
  chat_response = response.data.chat_response
  parts: list[types.Part] = []
  input_tokens = 0
  output_tokens = 0
  reasoning_tokens = 0

  if hasattr(chat_response, "usage"):
    usage = chat_response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
      reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

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
        if part.function_call is not None:
          part.function_call.id = getattr(fc, "id", "") or ""
        parts.append(part)

  return LlmResponse(
      content=types.Content(role="model", parts=parts),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=input_tokens,
          candidates_token_count=output_tokens,
          total_token_count=input_tokens + output_tokens,
          thoughts_token_count=reasoning_tokens or None,
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

      from google.adk.integrations.oci import OCIGenAILlm
      from google.adk.agents import LlmAgent

      agent = LlmAgent(
          model=OCIGenAILlm(
              model="google.gemini-2.0-flash-001",
              compartment_id="ocid1.compartment.oc1...",
          ),
          ...
      )

  Attributes:
    model: OCI model ID (e.g. ``google.gemini-2.0-flash-001``). Used as the
      ``model_id`` for on-demand serving. For dedicated serving, set
      ``endpoint_id`` instead; ``model`` is then informational only.
    endpoint_id: Dedicated endpoint OCID (``ocid1.generativeaiendpoint...``).
      When set, requests use ``DedicatedServingMode``; otherwise on-demand
      mode is used. Falls back to ``OCI_ENDPOINT_ID`` env var when not set.
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
    reasoning_effort: Reasoning-token budget for reasoning-capable models.
      One of ``"NONE"``, ``"MINIMAL"``, ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``,
      or ``None`` (default — let OCI pick). Honoured by GPT-5 family,
      Gemini 2.5, Grok reasoning variants, and Cohere Command-A-Reasoning;
      ignored by non-reasoning models. The single most impactful cost knob
      for reasoning models — ``"LOW"`` typically cuts reasoning-token spend
      5-10× vs the default.
  """

  model: str = "google.gemini-2.5-flash"
  endpoint_id: Optional[str] = None
  compartment_id: Optional[str] = None
  service_endpoint: Optional[str] = None
  auth_type: str = "API_KEY"
  auth_profile: str = "DEFAULT"
  auth_file_location: str = "~/.oci/config"
  max_tokens: int = 2048
  reasoning_effort: Optional[str] = None

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

  @cached_property
  def _oci_client(self) -> Any:
    return self._build_client(self._resolve_service_endpoint())

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

    messages = [_content_to_oci_message(c) for c in llm_request.contents or []]

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
    if llm_request.config and llm_request.config.tools:
      first_tool = llm_request.config.tools[0]
      if (
          isinstance(first_tool, types.Tool)
          and first_tool.function_declarations
      ):
        oci_tools = [
            _function_declaration_to_oci_tool(fn)
            for fn in first_tool.function_declarations
        ]

    chat_request_kwargs: dict[str, Any] = dict(
        api_format=oci_models.BaseChatRequest.API_FORMAT_GENERIC,
        messages=messages,
        max_tokens=self.max_tokens,
    )

    # Sampling and decoding parameters from llm_request.config.
    cfg = getattr(llm_request, "config", None)
    if cfg is not None:
      if cfg.max_output_tokens is not None:
        chat_request_kwargs["max_tokens"] = cfg.max_output_tokens
      if cfg.temperature is not None:
        chat_request_kwargs["temperature"] = cfg.temperature
      if cfg.top_p is not None:
        chat_request_kwargs["top_p"] = cfg.top_p
      if cfg.top_k is not None:
        chat_request_kwargs["top_k"] = int(cfg.top_k)
      if cfg.frequency_penalty is not None:
        chat_request_kwargs["frequency_penalty"] = cfg.frequency_penalty
      if cfg.presence_penalty is not None:
        chat_request_kwargs["presence_penalty"] = cfg.presence_penalty
      if cfg.seed is not None:
        chat_request_kwargs["seed"] = cfg.seed
      if cfg.stop_sequences:
        chat_request_kwargs["stop"] = list(cfg.stop_sequences)

      # Structured-output: response_schema (Pydantic / dict / google.genai
      # Schema) → JsonSchemaResponseFormat. response_mime_type alone (without
      # a schema) → JsonObjectResponseFormat (free-form JSON).
      response_format = _build_response_format(cfg, oci_models)
      if response_format is not None:
        chat_request_kwargs["response_format"] = response_format

    # Constructor-level reasoning_effort applies regardless of per-request cfg.
    if self.reasoning_effort is not None:
      chat_request_kwargs["reasoning_effort"] = self.reasoning_effort

    if oci_tools:
      chat_request_kwargs["tools"] = oci_tools
    if is_stream:
      chat_request_kwargs["is_stream"] = True
      chat_request_kwargs["stream_options"] = oci_models.StreamOptions(
          is_include_usage=True
      )

    return oci_models.ChatDetails(
        compartment_id=self._resolve_compartment_id(),
        serving_mode=self._build_serving_mode(oci_models),
        chat_request=oci_models.GenericChatRequest(**chat_request_kwargs),
    )

  def _build_serving_mode(self, oci_models: Any) -> Any:
    endpoint_id = self.endpoint_id or os.environ.get("OCI_ENDPOINT_ID")
    if endpoint_id:
      return oci_models.DedicatedServingMode(endpoint_id=endpoint_id)
    return oci_models.OnDemandServingMode(model_id=self.model)

  def _call_oci(self, llm_request: LlmRequest) -> Any:
    """Synchronous non-streaming OCI GenAI call, run in a thread pool."""
    chat_details = self._build_chat_details(llm_request, is_stream=False)
    logger.debug("Sending request to OCI GenAI: model=%s", self.model)
    return self._oci_client.chat(chat_details)

  def _call_oci_stream(self, llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Synchronous streaming call — collects all SSE event dicts in a thread.

    The OCI SDK wraps an SSE response in ``oci._vendor.sseclient.SSEClient``
    when ``is_stream=True`` is set on the request body.  Each event's
    ``data`` field is an OpenAI-compatible JSON chunk or the sentinel
    ``[DONE]``.
    """
    chat_details = self._build_chat_details(llm_request, is_stream=True)
    logger.debug("Sending streaming request to OCI GenAI: model=%s", self.model)
    response = self._oci_client.chat(chat_details)

    chunks: list[dict[str, Any]] = []
    try:
      for event in response.data.events():
        raw = getattr(event, "data", None)
        if not raw or raw.strip() == "[DONE]":
          break
        try:
          chunks.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
          logger.debug("Could not parse SSE event data: %r", raw)
    finally:
      close = getattr(response.data, "close", None)
      if callable(close):
        close()
    return chunks

  async def _generate_content_streaming(
      self, llm_request: LlmRequest
  ) -> AsyncGenerator[LlmResponse, None]:
    """Yield partial then final LlmResponse from an OCI SSE stream.

    The OCI SDK is synchronous, so every SSE chunk is collected in a background
    thread before any response is emitted. Partial responses are therefore not
    delivered incrementally.
    """
    chunks = await asyncio.to_thread(self._call_oci_stream, llm_request)

    text_acc: str = ""
    tool_acc: dict[int, dict[str, Any]] = {}
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    for chunk in chunks:
      # Usage chunk (camelCase per OCI GenAI /20231130/ schema).
      usage = chunk.get("usage")
      if usage:
        input_tokens = usage.get("promptTokens", 0) or 0
        output_tokens = usage.get("completionTokens", 0) or 0
        details = usage.get("completionTokensDetails") or {}
        reasoning_tokens = details.get("reasoningTokens", 0) or 0
        continue

      message = chunk.get("message")
      if not message:
        continue

      # Text content: list of {type: TEXT, text: ...} blocks.
      for block in message.get("content") or []:
        if block.get("type") == "TEXT" and block.get("text"):
          delta_text = block["text"]
          text_acc += delta_text
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text=delta_text)],
              ),
              partial=True,
          )

      # Tool calls: OCI emits the whole call in one chunk for Gemini, but
      # accumulate name/arguments defensively in case other providers split
      # them across events.
      for tc_idx, tc in enumerate(message.get("toolCalls") or []):
        idx = tc.get("index", tc_idx)
        if idx not in tool_acc:
          tool_acc[idx] = {"id": "", "name": "", "arguments": ""}
        if tc.get("id"):
          tool_acc[idx]["id"] = tc["id"]
        if tc.get("name"):
          tool_acc[idx]["name"] = tc["name"]
        if tc.get("arguments"):
          tool_acc[idx]["arguments"] += tc["arguments"]

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
      if part.function_call is not None:
        part.function_call.id = tc["id"]
      all_parts.append(part)

    yield LlmResponse(
        content=types.Content(role="model", parts=all_parts),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=input_tokens + output_tokens,
            thoughts_token_count=reasoning_tokens or None,
        ),
        partial=False,
    )
