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

"""Anthropic integration for Claude models."""

from __future__ import annotations

import base64
import copy
import dataclasses
from functools import cached_property
import json
import logging
import os
import re
from typing import Any
from typing import AsyncGenerator
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
import warnings

from anthropic import AsyncAnthropic
from anthropic import AsyncAnthropicVertex
from anthropic import NOT_GIVEN
from anthropic import NotGiven
from anthropic import types as anthropic_types
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from ..utils._google_client_headers import get_tracking_headers
from .base_llm import BaseLlm
from .interactions_utils import extract_system_instruction
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

__all__ = ["AnthropicLlm", "Claude", "AnthropicGenerateContentConfig"]

logger = logging.getLogger("google_adk." + __name__)


@dataclasses.dataclass
class _ToolUseAccumulator:
  """Accumulates streamed tool_use content block data."""

  id: str
  name: str
  args_json: str


@dataclasses.dataclass
class _ThinkingAccumulator:
  """Accumulates streamed thinking content block data."""

  thinking: str
  signature: str


def _build_anthropic_thinking_param(
    config: Optional[types.GenerateContentConfig],
) -> Union[
    anthropic_types.ThinkingConfigEnabledParam,
    anthropic_types.ThinkingConfigDisabledParam,
    anthropic_types.ThinkingConfigAdaptiveParam,
    NotGiven,
]:
  """Maps genai ThinkingConfig to Anthropic's thinking parameter.

  Per ``google.genai.types.ThinkingConfig``, ``thinking_budget`` semantics are:
    * ``None``: not specified; the genai default is model-dependent. Anthropic
      requires an explicit choice whenever thinking is configured, so we
      surface this as a ``ValueError`` to keep the developer's intent
      explicit (mirroring the Anthropic API).
    * ``0``: thinking is DISABLED (``thinking.type: "disabled"``).
    * negative (e.g. ``-1`` AUTOMATIC): maps to Anthropic's adaptive thinking
      (``thinking.type: "adaptive"``). The model picks the depth itself
      (controlled by the separate ``output_config.effort`` parameter when
      set). REQUIRED for Claude Opus 4.7 and later models that reject
      ``"enabled"`` with a 400 error; also recommended for Opus 4.6 and
      Sonnet 4.6 where ``"enabled"`` is deprecated.
    * positive int: budget in tokens for legacy manual mode
      (``thinking.type: "enabled"``; Anthropic requires ``>= 1024`` and
      ``< max_tokens``; validation is delegated to the Anthropic API so the
      caller gets the canonical error message). Rejected by Claude Opus 4.7
      -- callers targeting 4.7+ must use a negative value (adaptive) or
      ``0`` (disabled).

  Args:
    config: Optional GenerateContentConfig object.

  Returns:
    Mapped thinking parameter or NotGiven.
  """
  if not config or not config.thinking_config:
    return NOT_GIVEN

  thinking_budget = config.thinking_config.thinking_budget

  if thinking_budget is None:
    raise ValueError(
        "thinking_budget must be set explicitly when ThinkingConfig is"
        " provided for Anthropic models. Use 0 to disable thinking, -1 for"
        " adaptive (model-chosen depth), or a positive integer (>= 1024)"
        " for manual budgeting."
    )

  if thinking_budget == 0:
    return anthropic_types.ThinkingConfigDisabledParam(type="disabled")

  if thinking_budget < 0:
    # genai AUTOMATIC (-1) and any other negative value map to Anthropic
    # adaptive thinking. Required for Claude Opus 4.7 (which returns a 400
    # error for ``"enabled"``) and recommended for Opus 4.6 / Sonnet 4.6
    # where ``"enabled"`` is deprecated. Adaptive does not accept a budget;
    # depth is controlled by the model itself (or by the separate
    # ``output_config.effort`` parameter when set).
    return anthropic_types.ThinkingConfigAdaptiveParam(type="adaptive")

  return anthropic_types.ThinkingConfigEnabledParam(
      type="enabled",
      budget_tokens=thinking_budget,
  )


class AnthropicGenerateContentConfig(types.GenerateContentConfig):
  """Configuration options for Anthropic Claude content generation.

  This specialized configuration class is the recommended way to
  configure reasoning and extended thinking for newer Claude models.

  Attributes:
    effort: The reasoning effort level for adaptive extended thinking. Set
      directly to guide the reasoning depth ("low", "medium", "high", "xhigh",
      "max"). This is the preferred alternative to the deprecated manual
      `thinking_budget` on newer Claude models.
  """

  effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = Field(
      default=None,
      description=(
          "Configures the Claude-specific reasoning effort level for adaptive"
          " extended thinking. This is the recommended, future-proof way to"
          " control reasoning depth on newer Claude models."
      ),
  )

  @model_validator(mode="after")
  def validate_no_thinking_level(self) -> "AnthropicGenerateContentConfig":
    """Ensures thinking_level is not configured on Anthropic-specific config."""

    if self.thinking_config and self.thinking_config.thinking_level is not None:
      raise ValueError(
          "thinking_level is not supported in AnthropicGenerateContentConfig. "
          "Use the `effort` field directly to configure reasoning effort."
      )
    return self


def _build_effort_param(
    config: Optional[types.GenerateContentConfig],
) -> Optional[str]:
  """Extracts Anthropic's effort parameter from the configuration.

  To configure a specific reasoning effort level for Anthropic models,
  callers must use ``google.adk.models.AnthropicGenerateContentConfig`` and
  set the ``effort`` field directly.
  Using the standard ``thinking_config.thinking_level`` is explicitly
  unsupported because the standard `ThinkingLevel` enum (4 levels) cannot map
  consistently to Anthropic's 5 effort levels
  ("low", "medium", "high", "xhigh", "max").

  Any attempt to set `thinking_level` will not be passed to the model and will
  log a warning.

  If `effort` is not set, we return `None`.
  If `effort` and `thinking_level` are both set, `effort` takes precedence.

  Args:
    config: Optional GenerateContentConfig object.

  Returns:
    The effort level string (e.g., "xhigh") if specified via
    AnthropicGenerateContentConfig, or None.
  """
  if not config:
    return None

  if isinstance(config, AnthropicGenerateContentConfig) and config.effort:
    return config.effort

  # If effort is not set, but thinking_level is, log a warning and ignore it.
  if config.thinking_config and config.thinking_config.thinking_level:
    warnings.warn(
        "Standard thinking_config.thinking_level is not supported for Anthropic"
        " models and will be ignored. Use AnthropicGenerateContentConfig and"
        " set the `effort` field directly to configure reasoning effort.",
        category=UserWarning,
        stacklevel=4,
    )

  return None


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


def _is_image_part(part: types.Part) -> bool:
  return (
      part.inline_data
      and part.inline_data.mime_type
      and part.inline_data.mime_type.startswith("image")
  )


def _is_pdf_part(part: types.Part) -> bool:
  return (
      part.inline_data
      and part.inline_data.mime_type
      and part.inline_data.mime_type.split(";")[0].strip() == "application/pdf"
  )


class _ToolUseIdSanitizer:
  """Maps invalid tool_use IDs to deterministic fallbacks.

  Reuse one instance per conversation so a tool_use and its paired
  tool_result with the same invalid source ID get matching outputs.
  """

  def __init__(self) -> None:
    self._mapping: dict[str, str] = {}
    self._next_fallback: int = 0

  def sanitize(self, tool_id: str | None) -> str:
    if tool_id and re.fullmatch(r"[a-zA-Z0-9_-]+", tool_id):
      return tool_id
    key = tool_id or ""
    if key not in self._mapping:
      self._mapping[key] = f"toolu_fallback_{self._next_fallback}"
      self._next_fallback += 1
    return self._mapping[key]


def _part_to_message_block(
    part: types.Part,
    sanitizer: _ToolUseIdSanitizer,
) -> Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ThinkingBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.DocumentBlockParam,
    anthropic_types.ToolUseBlockParam,
    anthropic_types.ToolResultBlockParam,
]:
  if part.thought and part.text:
    signature = ""
    if part.thought_signature:
      signature = part.thought_signature.decode("utf-8")
    return anthropic_types.ThinkingBlockParam(
        type="thinking",
        thinking=part.text,
        signature=signature,
    )
  if part.thought and part.thought_signature:
    # Redacted thinking: no plaintext, only the encrypted blob produced by
    # content_block_to_part for round-tripping back to Claude.
    return anthropic_types.RedactedThinkingBlockParam(
        type="redacted_thinking",
        data=part.thought_signature.decode("utf-8"),
    )
  if part.text:
    return anthropic_types.TextBlockParam(text=part.text, type="text")
  elif part.function_call:
    assert part.function_call.name

    return anthropic_types.ToolUseBlockParam(
        id=sanitizer.sanitize(part.function_call.id),
        name=part.function_call.name,
        input=part.function_call.args,
        type="tool_use",
    )
  elif part.function_response:
    content = ""
    response_data = part.function_response.response

    if (
        "content" in response_data
        and isinstance(response_data["content"], list)
        and response_data["content"]
    ):
      content_items = []
      for item in response_data["content"]:
        if isinstance(item, dict):
          if item.get("type") == "text" and "text" in item:
            content_items.append(item["text"])
          else:
            content_items.append(str(item))
        else:
          content_items.append(str(item))
      content = "\n".join(content_items) if content_items else ""
    elif (
        "content" in response_data
        and isinstance(response_data["content"], str)
        and response_data["content"]
    ):
      content = response_data["content"]
    # We serialize to str here
    # SDK ref: anthropic.types.tool_result_block_param
    # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_result_block_param.py
    elif "result" in response_data and response_data["result"] is not None:
      result = response_data["result"]
      if isinstance(result, (dict, list)):
        content = json.dumps(result)
      else:
        content = str(result)
    elif response_data:
      # Fallback: serialize the entire response dict as JSON so that tools
      # returning arbitrary key structures (e.g. load_skill returning
      # {"skill_name", "instructions", "frontmatter"}) are not silently
      # dropped.
      content = json.dumps(response_data)

    return anthropic_types.ToolResultBlockParam(
        tool_use_id=sanitizer.sanitize(part.function_response.id),
        type="tool_result",
        content=content,
        is_error=False,
    )
  elif _is_image_part(part):
    data = base64.b64encode(part.inline_data.data).decode()
    return anthropic_types.ImageBlockParam(
        type="image",
        source=dict(
            type="base64", media_type=part.inline_data.mime_type, data=data
        ),
    )
  elif _is_pdf_part(part):
    data = base64.b64encode(part.inline_data.data).decode()
    return anthropic_types.DocumentBlockParam(
        type="document",
        source=dict(
            type="base64", media_type=part.inline_data.mime_type, data=data
        ),
    )
  elif part.executable_code:
    return anthropic_types.TextBlockParam(
        type="text",
        text="Code:```python\n" + part.executable_code.code + "\n```",
    )
  elif part.code_execution_result:
    return anthropic_types.TextBlockParam(
        text="Execution Result:```code_output\n"
        + part.code_execution_result.output
        + "\n```",
        type="text",
    )

  raise NotImplementedError(f"Not supported yet: {part}")


def _content_to_message_param(
    content: types.Content,
    sanitizer: _ToolUseIdSanitizer,
) -> anthropic_types.MessageParam:
  message_block = []
  for part in content.parts or []:
    # Image data is not supported in Claude for assistant turns.
    if content.role != "user" and _is_image_part(part):
      logger.warning(
          "Image data is not supported in Claude for assistant turns."
      )
      continue

    # PDF data is not supported in Claude for assistant turns.
    if content.role != "user" and _is_pdf_part(part):
      logger.warning("PDF data is not supported in Claude for assistant turns.")
      continue

    message_block.append(_part_to_message_block(part, sanitizer))

  return {
      "role": to_claude_role(content.role),
      "content": message_block,
  }


def part_to_message_block(
    part: types.Part,
) -> Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.DocumentBlockParam,
    anthropic_types.ToolUseBlockParam,
    anthropic_types.ToolResultBlockParam,
]:
  return _part_to_message_block(part, _ToolUseIdSanitizer())


def content_to_message_param(
    content: types.Content,
) -> anthropic_types.MessageParam:
  return _content_to_message_param(content, _ToolUseIdSanitizer())


def content_block_to_part(
    content_block: anthropic_types.ContentBlock,
) -> types.Part:
  """Converts an Anthropic content block to a genai Part."""
  if isinstance(content_block, anthropic_types.ThinkingBlock):
    part = types.Part(text=content_block.thinking, thought=True)
    if content_block.signature:
      part.thought_signature = content_block.signature.encode("utf-8")
    return part
  if isinstance(content_block, anthropic_types.RedactedThinkingBlock):
    # Preserve the encrypted blob so it can round-trip back to Claude in
    # the next turn; required to keep the model's reasoning chain intact.
    return types.Part(
        thought=True,
        thought_signature=content_block.data.encode("utf-8"),
    )
  if isinstance(content_block, anthropic_types.TextBlock):
    return types.Part.from_text(text=content_block.text)
  if isinstance(content_block, anthropic_types.ToolUseBlock):
    assert isinstance(content_block.input, dict)
    part = types.Part.from_function_call(
        name=content_block.name, args=content_block.input
    )
    part.function_call.id = content_block.id
    return part
  raise NotImplementedError(
      f"Unsupported content block type: {type(content_block)}"
  )


def _extract_cached_token_count(usage: Any) -> int | None:
  """Returns Anthropic cache-read tokens, the analog of cached_content tokens."""
  cached = getattr(usage, "cache_read_input_tokens", None)
  return cached if isinstance(cached, int) else None


def message_to_generate_content_response(
    message: anthropic_types.Message,
) -> LlmResponse:
  logger.info("Received response from Claude.")
  logger.debug(
      "Claude response: %s",
      message.model_dump_json(indent=2, exclude_none=True),
  )

  parts = [content_block_to_part(cb) for cb in message.content]

  return LlmResponse(
      content=types.Content(
          role="model",
          parts=parts,
      ),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=message.usage.input_tokens,
          candidates_token_count=message.usage.output_tokens,
          total_token_count=(
              message.usage.input_tokens + message.usage.output_tokens
          ),
          cached_content_token_count=_extract_cached_token_count(message.usage),
      ),
      # TODO: Deal with these later.
      # finish_reason=to_google_genai_finish_reason(message.stop_reason),
  )


def _update_type_string(value: Any):
  """Lowercases nested JSON schema type strings for Anthropic compatibility."""
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


def function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> anthropic_types.ToolParam:
  """Converts a function declaration to an Anthropic tool param."""
  assert function_declaration.name

  # Use parameters_json_schema if available, otherwise convert from parameters
  if function_declaration.parameters_json_schema:
    input_schema = copy.deepcopy(function_declaration.parameters_json_schema)
    _update_type_string(input_schema)
  else:
    properties = {}
    required_params = []
    if function_declaration.parameters:
      if function_declaration.parameters.properties:
        for key, value in function_declaration.parameters.properties.items():
          properties[key] = value.model_dump(by_alias=True, exclude_none=True)
      if function_declaration.parameters.required:
        required_params = function_declaration.parameters.required

    input_schema = {
        "type": "object",
        "properties": properties,
    }
    if required_params:
      input_schema["required"] = required_params
    _update_type_string(input_schema)

  return anthropic_types.ToolParam(
      name=function_declaration.name,
      description=function_declaration.description or "",
      input_schema=input_schema,
  )


class AnthropicLlm(BaseLlm):
  """Integration with Claude models via the Anthropic API.

  Note:
    Anthropic Claude supports 5 distinct effort levels ("low", "medium",
    "high", "xhigh", "max") while the standard `ThinkingLevel` enum defines 4
    levels (MINIMAL, LOW, MEDIUM, HIGH), the standard
    `thinking_config.thinking_level` is not supported for Anthropic models.
    To configure thinking effort, user must use `AnthropicGenerateContentConfig`
    and set its `effort` field directly (e.g., `effort="xhigh"`).

  Attributes:
    model: The name of the Claude model.
    max_tokens: The maximum number of tokens to generate.
  """

  model: str = "claude-sonnet-4-20250514"
  max_tokens: int = 8192

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r"claude-3-.*", r"claude-.*-4.*"]

  def _resolve_model_name(self, model: Optional[str]) -> str:
    if not model:
      return self.model
    if model.startswith("projects/"):
      match = re.search(
          r"projects/[^/]+/locations/[^/]+/(?:publishers/anthropic/models|endpoints)/([^/:]+)",
          model,
      )
      if match:
        return match.group(1)
    return model

  def _build_anthropic_kwargs(
      self,
      llm_request: LlmRequest,
      messages: list[anthropic_types.MessageParam],
      tools: Union[Iterable[anthropic_types.ToolUnionParam], NotGiven],
      tool_choice: Union[anthropic_types.ToolChoiceParam, NotGiven],
      thinking: Union[
          anthropic_types.ThinkingConfigEnabledParam,
          anthropic_types.ThinkingConfigDisabledParam,
          anthropic_types.ThinkingConfigAdaptiveParam,
          NotGiven,
      ],
  ) -> dict[str, Any]:
    system = NOT_GIVEN
    if llm_request.config:
      system_str = extract_system_instruction(llm_request.config)
      if system_str:
        system = system_str

    model_to_use = self._resolve_model_name(llm_request.model)
    kwargs = {
        "model": model_to_use,
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "thinking": thinking,
    }

    effort = _build_effort_param(llm_request.config)
    if effort:
      kwargs["output_config"] = {"effort": effort}

    # Determine if thinking is enabled to avoid parameter conflicts.
    thinking_enabled = False
    if thinking is not NOT_GIVEN and thinking is not None:
      if isinstance(thinking, dict):
        thinking_enabled = thinking.get("type") in ["enabled", "adaptive"]

    exclude_sampling = thinking_enabled or (effort is not None)

    if llm_request.config:
      # Models released after Claude Opus 4.6 do not support setting
      # temperature, top_k, or top_p when thinking is enabled or effort is set.
      if not exclude_sampling:
        if llm_request.config.temperature is not None:
          kwargs["temperature"] = llm_request.config.temperature
        if llm_request.config.top_p is not None:
          kwargs["top_p"] = llm_request.config.top_p
        if llm_request.config.top_k is not None:
          kwargs["top_k"] = int(llm_request.config.top_k)
      else:
        if (
            llm_request.config.temperature is not None
            or llm_request.config.top_p is not None
            or llm_request.config.top_k is not None
        ):
          warnings.warn(
              "Sampling parameters (temperature, top_p, top_k) are ignored "
              "because thinking/effort is enabled.",
              category=UserWarning,
              stacklevel=3,
          )

      if llm_request.config.stop_sequences:
        kwargs["stop_sequences"] = llm_request.config.stop_sequences

      if llm_request.config.max_output_tokens is not None:
        kwargs["max_tokens"] = llm_request.config.max_output_tokens
      else:
        kwargs["max_tokens"] = self.max_tokens
    else:
      kwargs["max_tokens"] = self.max_tokens

    return kwargs

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    sanitizer = _ToolUseIdSanitizer()
    messages = [
        _content_to_message_param(content, sanitizer)
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
    thinking = _build_anthropic_thinking_param(llm_request.config)

    if not stream:
      kwargs = self._build_anthropic_kwargs(
          llm_request, messages, tools, tool_choice, thinking
      )
      message = await self._anthropic_client.messages.create(**kwargs)
      yield message_to_generate_content_response(message)
    else:
      async for response in self._generate_content_streaming(
          llm_request, messages, tools, tool_choice, thinking
      ):
        yield response

  async def _generate_content_streaming(
      self,
      llm_request: LlmRequest,
      messages: list[anthropic_types.MessageParam],
      tools: Union[Iterable[anthropic_types.ToolUnionParam], NotGiven],
      tool_choice: Union[anthropic_types.ToolChoiceParam, NotGiven],
      thinking: Union[
          anthropic_types.ThinkingConfigEnabledParam,
          anthropic_types.ThinkingConfigDisabledParam,
          anthropic_types.ThinkingConfigAdaptiveParam,
          NotGiven,
      ] = NOT_GIVEN,
  ) -> AsyncGenerator[LlmResponse, None]:
    """Handles streaming responses from Anthropic models.

    Args:
      llm_request: LlmRequest containing configurations and contents.
      messages: List of formatted Anthropic messages.
      tools: Optional tool configurations.
      tool_choice: Optional tool choice setting.
      thinking: Optional thinking details.

    Yields:
      Partial LlmResponse objects as content arrives, followed by
      a final aggregated LlmResponse with all content.
    """
    kwargs = self._build_anthropic_kwargs(
        llm_request, messages, tools, tool_choice, thinking
    )
    raw_stream = await self._anthropic_client.messages.create(
        stream=True,
        **kwargs,
    )

    # Track content blocks being built during streaming.
    # Each entry maps a block index to its accumulated state.
    text_blocks: dict[int, str] = {}
    tool_use_blocks: dict[int, _ToolUseAccumulator] = {}
    thinking_blocks: dict[int, _ThinkingAccumulator] = {}
    redacted_thinking_blocks: dict[int, str] = {}
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens: int | None = None

    async for event in raw_stream:
      if event.type == "message_start":
        input_tokens = event.message.usage.input_tokens
        output_tokens = event.message.usage.output_tokens
        cached_input_tokens = _extract_cached_token_count(event.message.usage)

      elif event.type == "content_block_start":
        block = event.content_block
        if isinstance(block, anthropic_types.ThinkingBlock):
          thinking_blocks[event.index] = _ThinkingAccumulator(
              thinking=block.thinking,
              signature=block.signature,
          )
        elif isinstance(block, anthropic_types.RedactedThinkingBlock):
          # Redacted blocks arrive fully formed at start; no deltas follow.
          redacted_thinking_blocks[event.index] = block.data
        elif isinstance(block, anthropic_types.TextBlock):
          text_blocks[event.index] = block.text
        elif isinstance(block, anthropic_types.ToolUseBlock):
          tool_use_blocks[event.index] = _ToolUseAccumulator(
              id=block.id,
              name=block.name,
              args_json="",
          )

      elif event.type == "content_block_delta":
        delta = event.delta
        if isinstance(delta, anthropic_types.ThinkingDelta):
          thinking_blocks.setdefault(
              event.index,
              _ThinkingAccumulator(thinking="", signature=""),
          )
          thinking_blocks[event.index].thinking += delta.thinking
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part(text=delta.thinking, thought=True)],
              ),
              partial=True,
          )
        elif isinstance(delta, anthropic_types.TextDelta):
          text_blocks.setdefault(event.index, "")
          text_blocks[event.index] += delta.text
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text=delta.text)],
              ),
              partial=True,
          )
        elif isinstance(delta, anthropic_types.InputJSONDelta):
          if event.index in tool_use_blocks:
            tool_use_blocks[event.index].args_json += delta.partial_json

      elif event.type == "message_delta":
        output_tokens = event.usage.output_tokens

    # Build the final aggregated response with all content.
    all_parts: list[types.Part] = []
    all_indices = sorted(
        set(
            list(thinking_blocks.keys())
            + list(redacted_thinking_blocks.keys())
            + list(text_blocks.keys())
            + list(tool_use_blocks.keys())
        )
    )
    for idx in all_indices:
      if idx in thinking_blocks:
        acc = thinking_blocks[idx]
        part = types.Part(text=acc.thinking, thought=True)
        if acc.signature:
          part.thought_signature = acc.signature.encode("utf-8")
        all_parts.append(part)
      if idx in redacted_thinking_blocks:
        all_parts.append(
            types.Part(
                thought=True,
                thought_signature=redacted_thinking_blocks[idx].encode("utf-8"),
            )
        )
      if idx in text_blocks:
        all_parts.append(types.Part.from_text(text=text_blocks[idx]))
      if idx in tool_use_blocks:
        acc = tool_use_blocks[idx]
        args = json.loads(acc.args_json) if acc.args_json else {}
        part = types.Part.from_function_call(name=acc.name, args=args)
        part.function_call.id = acc.id
        all_parts.append(part)

    yield LlmResponse(
        content=types.Content(role="model", parts=all_parts),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=input_tokens + output_tokens,
            cached_content_token_count=cached_input_tokens,
        ),
        partial=False,
    )

  @cached_property
  def _anthropic_client(self) -> AsyncAnthropic:
    return AsyncAnthropic()


class Claude(AnthropicLlm):
  """Integration with Claude models served from Vertex AI.

  Note:
    Because Anthropic Claude supports 5 distinct effort levels ("low", "medium",
    "high", "xhigh", "max") while the standard `ThinkingLevel` enum defines 4
    levels (MINIMAL, LOW, MEDIUM, HIGH), the standard
    `thinking_config.thinking_level` is not supported for Anthropic models.

    To configure thinking effort, user must use `AnthropicGenerateContentConfig`
    and set its `effort` field directly (e.g., `effort="xhigh"`).

  Attributes:
    model: The name of the Claude model.
    max_tokens: The maximum number of tokens to generate.
  """

  model: str = "claude-3-5-sonnet-v2@20241022"

  @cached_property
  @override
  def _anthropic_client(self) -> AsyncAnthropicVertex:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")

    if self.model.startswith("projects/"):
      match = re.search(
          r"projects/([^/]+)/locations/([^/]+)/",
          self.model,
      )
      if match:
        project_id = match.group(1)
        location = match.group(2)

    if not project_id or not location:
      raise ValueError(
          "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for using"
          " Anthropic on Vertex."
      )

    return AsyncAnthropicVertex(
        project_id=project_id,
        region=location,
        default_headers=get_tracking_headers(),
    )
