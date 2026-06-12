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

"""OCI Generative AI integration via the OpenAI-compatible v1 transport.

This module exposes ``OCIGenAIOpenAILlm`` — a ``BaseLlm`` that talks to OCI
Generative AI's documented OpenAI-compatible Chat Completions endpoint at
``/20231130/actions/v1/chat/completions`` using the ``openai`` Python SDK.
It is a thinner alternative to :class:`OCIGenAILlm` (which uses OCI's native
inference SDK) for users who prefer the OpenAI-style transport.

Reference docs:
  - https://docs.oracle.com/en-us/iaas/Content/generative-ai/openai-compatible-api.htm
  - https://github.com/oracle-samples/oci-openai
"""

from __future__ import annotations

import asyncio
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

if not TYPE_CHECKING and importlib.util.find_spec("openai") is None:
  raise ImportError(
      "OCI OpenAI-compatible transport requires the openai package: "
      "pip install google-adk[oci]"
  )

from .base_llm import BaseLlm
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest


__all__ = ["OCIGenAIOpenAILlm"]

logger = logging.getLogger("google_adk." + __name__)


def _content_to_openai_messages(
    contents: list[types.Content],
    system_instruction: Optional[str],
) -> list[dict[str, Any]]:
  """Convert google.genai Content list to OpenAI chat message dicts.

  Only text parts are emitted in this transport; multimodal (inline_data /
  file_data) and tool-call round-trips are forwarded as text content when
  present and silently dropped when not stringifiable. Callers who need
  full multimodal/tool fidelity should use OCIGenAILlm instead.
  """
  messages: list[dict[str, Any]] = []
  if system_instruction:
    messages.append({"role": "system", "content": system_instruction})

  for content in contents or []:
    role = content.role or "user"
    if role == "model":
      role = "assistant"

    text_buf: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_response: Optional[tuple[str, str]] = None

    for part in content.parts or []:
      if part.text:
        text_buf.append(part.text)
      elif part.function_call:
        tool_calls.append({
            "id": part.function_call.id or "",
            "type": "function",
            "function": {
                "name": part.function_call.name,
                "arguments": json.dumps(part.function_call.args or {}),
            },
        })
      elif part.function_response:
        # Tool result — convert to a 'tool' role message later.
        tool_response = (
            part.function_response.id or part.function_response.name or "",
            json.dumps(part.function_response.response or {}),
        )

    if tool_response is not None:
      tool_id, tool_text = tool_response
      messages.append({
          "role": "tool",
          "tool_call_id": tool_id,
          "content": tool_text,
      })
      continue

    msg: dict[str, Any] = {"role": role, "content": "".join(text_buf)}
    if tool_calls:
      msg["tool_calls"] = tool_calls
    messages.append(msg)

  return messages


def _tools_to_openai(
    tools: Optional[list[types.Tool]],
) -> Optional[list[dict[str, Any]]]:
  """Convert google.genai Tool list to OpenAI function-tool dicts."""
  if not tools:
    return None
  out: list[dict[str, Any]] = []
  for tool in tools:
    for decl in tool.function_declarations or []:
      params = decl.parameters
      params_dict: dict[str, Any]
      if params is None:
        params_dict = {"type": "object", "properties": {}}
      elif hasattr(params, "to_json_dict"):
        params_dict = params.to_json_dict()
      else:
        params_dict = dict(params)  # already a dict
      out.append({
          "type": "function",
          "function": {
              "name": decl.name,
              "description": decl.description or "",
              "parameters": params_dict,
          },
      })
  return out or None


def _openai_response_to_llm_response(resp: Any) -> LlmResponse:
  """Convert a non-streaming openai ChatCompletion to LlmResponse."""
  choice = resp.choices[0]
  msg = choice.message
  parts: list[types.Part] = []
  if msg.content:
    parts.append(types.Part.from_text(text=msg.content))
  for tc in msg.tool_calls or []:
    try:
      args = json.loads(tc.function.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
      args = {}
    parts.append(
        types.Part.from_function_call(
            name=tc.function.name,
            args=args,
        )
    )

  usage = None
  if resp.usage:
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=resp.usage.prompt_tokens,
        candidates_token_count=resp.usage.completion_tokens,
        total_token_count=resp.usage.total_tokens,
    )

  return LlmResponse(
      content=types.Content(role="model", parts=parts),
      usage_metadata=usage,
  )


class OCIGenAIOpenAILlm(BaseLlm):
  """OCI Generative AI via the OpenAI-compatible v1 transport.

  Targets ``/20231130/actions/v1/chat/completions`` on the OCI Generative
  AI inference endpoint. Two authentication modes:

  - ``BEARER_TOKEN`` — OCI Generative AI API key (introduced 2026-01-21),
    sent as a plain ``Authorization: Bearer …`` header. Simplest path.
  - ``API_KEY`` (default) — OCI IAM request signing via ``~/.oci/config``.

  Example::

      from google.adk.models.oci_genai_openai_llm import OCIGenAIOpenAILlm

      # Bearer token (simplest)
      llm = OCIGenAIOpenAILlm(
          model="google.gemini-2.5-flash",
          auth_type="BEARER_TOKEN",
          api_key="<oci-genai-api-key>",
          compartment_id="ocid1.compartment.oc1..xxx",
      )

      # IAM request signing (default, uses ~/.oci/config)
      llm = OCIGenAIOpenAILlm(
          model="google.gemini-2.5-flash",
          compartment_id="ocid1.compartment.oc1..xxx",
      )

  Use :class:`OCIGenAILlm` (native SDK) when you need full multimodal,
  tool-call streaming fidelity, or Cohere-format chat history.
  """

  model: str = "google.gemini-2.5-flash"
  compartment_id: Optional[str] = None
  region: str = "us-chicago-1"
  service_endpoint: Optional[str] = None
  auth_type: str = "API_KEY"
  api_key: Optional[str] = None
  auth_profile: str = "DEFAULT"
  auth_file_location: str = "~/.oci/config"
  max_tokens: int = 2048

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    # Not auto-routable: users instantiate this class explicitly when they
    # want the OpenAI-compat transport. OCIGenAILlm owns the regex routing
    # for OCI model ids.
    return []

  @override
  async def generate_content_async(
      self,
      llm_request: LlmRequest,
      stream: bool = False,
  ) -> AsyncGenerator[LlmResponse, None]:
    if stream:
      async for response in self._stream(llm_request):
        yield response
    else:
      response = await asyncio.to_thread(self._chat, llm_request)
      yield _openai_response_to_llm_response(response)

  # ------------------------------------------------------------------
  # Client + request construction
  # ------------------------------------------------------------------

  def _resolve_compartment_id(self) -> str:
    cid = self.compartment_id or os.environ.get("OCI_COMPARTMENT_ID")
    if not cid:
      raise ValueError(
          "compartment_id must be set on OCIGenAIOpenAILlm or via the"
          " OCI_COMPARTMENT_ID environment variable."
      )
    return cid

  def _resolve_base_url(self) -> str:
    return (
        self.service_endpoint
        or os.environ.get("OCI_SERVICE_ENDPOINT")
        or (
            f"https://inference.generativeai.{self.region}.oci.oraclecloud.com"
            "/20231130/actions/v1"
        )
    )

  def _build_client(self) -> Any:
    import openai

    compartment_id = self._resolve_compartment_id()
    base_url = self._resolve_base_url()
    default_headers = {"opc-compartment-id": compartment_id}
    auth_type = self.auth_type.upper()

    if auth_type == "BEARER_TOKEN":
      key = self.api_key or os.environ.get("OCI_GENAI_API_KEY")
      if not key:
        raise ValueError(
            "auth_type='BEARER_TOKEN' requires api_key=… or the"
            " OCI_GENAI_API_KEY environment variable to be set."
        )
      return openai.OpenAI(
          api_key=key,
          base_url=base_url,
          default_headers=default_headers,
      )

    # IAM modes — wrap with httpx auth that signs each request.
    import httpx

    signer = self._build_signer(auth_type)

    class _OCIAuth(httpx.Auth):

      def __init__(self, signer):
        self._signer = signer

      def auth_flow(self, request: httpx.Request):
        import requests as _requests

        prep = _requests.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=request.content,
        ).prepare()
        self._signer(prep)
        for k, v in prep.headers.items():
          request.headers[k] = v
        yield request

    return openai.OpenAI(
        api_key="oci",  # placeholder; real auth via the http_client signer
        base_url=base_url,
        default_headers=default_headers,
        http_client=httpx.Client(auth=_OCIAuth(signer)),
    )

  def _build_signer(self, auth_type: str) -> Any:
    import oci

    if auth_type == "INSTANCE_PRINCIPAL":
      return oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    if auth_type == "RESOURCE_PRINCIPAL":
      return oci.auth.signers.get_resource_principals_signer()

    config = oci.config.from_file(
        file_location=self.auth_file_location,
        profile_name=self.auth_profile,
    )
    config["region"] = self.region
    if not self.compartment_id:
      self.compartment_id = config["tenancy"]
    return oci.Signer(
        tenancy=config["tenancy"],
        user=config["user"],
        fingerprint=config["fingerprint"],
        private_key_file_location=config.get("key_file"),
        private_key_content=config.get("key_content"),
    )

  def _build_create_kwargs(
      self, llm_request: LlmRequest, stream: bool
  ) -> dict[str, Any]:
    cfg = llm_request.config or types.GenerateContentConfig()
    messages = _content_to_openai_messages(
        llm_request.contents or [],
        system_instruction=getattr(cfg, "system_instruction", None),
    )
    kwargs: dict[str, Any] = {
        "model": llm_request.model or self.model,
        "messages": messages,
        "stream": stream,
    }
    if cfg.max_output_tokens is not None:
      kwargs["max_tokens"] = cfg.max_output_tokens
    elif self.max_tokens:
      kwargs["max_tokens"] = self.max_tokens
    if cfg.temperature is not None:
      kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
      kwargs["top_p"] = cfg.top_p
    if cfg.stop_sequences:
      kwargs["stop"] = list(cfg.stop_sequences)
    tools = _tools_to_openai(cfg.tools)
    if tools:
      kwargs["tools"] = tools
    if stream:
      kwargs["stream_options"] = {"include_usage": True}
    return kwargs

  def _chat(self, llm_request: LlmRequest) -> Any:
    client = self._build_client()
    kwargs = self._build_create_kwargs(llm_request, stream=False)
    return client.chat.completions.create(**kwargs)

  async def _stream(
      self, llm_request: LlmRequest
  ) -> AsyncGenerator[LlmResponse, None]:
    chunks = await asyncio.to_thread(self._collect_stream, llm_request)
    text_acc = ""
    usage = None
    tool_acc: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
      if chunk.usage:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=chunk.usage.prompt_tokens,
            candidates_token_count=chunk.usage.completion_tokens,
            total_token_count=chunk.usage.total_tokens,
        )
        continue
      if not chunk.choices:
        continue
      delta = chunk.choices[0].delta
      if delta is None:
        continue
      if delta.content:
        text_acc += delta.content
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=delta.content)],
            ),
            partial=True,
        )
      for tc in delta.tool_calls or []:
        idx = tc.index
        slot = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
        if tc.id:
          slot["id"] = tc.id
        if tc.function and tc.function.name:
          slot["name"] = tc.function.name
        if tc.function and tc.function.arguments:
          slot["args"] += tc.function.arguments

    final_parts: list[types.Part] = []
    if text_acc:
      final_parts.append(types.Part.from_text(text=text_acc))
    for slot in tool_acc.values():
      try:
        args = json.loads(slot["args"] or "{}")
      except (json.JSONDecodeError, TypeError):
        args = {}
      final_parts.append(
          types.Part.from_function_call(name=slot["name"], args=args)
      )

    yield LlmResponse(
        content=types.Content(role="model", parts=final_parts),
        usage_metadata=usage,
    )

  def _collect_stream(self, llm_request: LlmRequest) -> list[Any]:
    client = self._build_client()
    kwargs = self._build_create_kwargs(llm_request, stream=True)
    return list(client.chat.completions.create(**kwargs))
