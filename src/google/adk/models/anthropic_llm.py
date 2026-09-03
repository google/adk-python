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
from typing import cast
from typing import get_args
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import Union
import warnings

from anthropic import AsyncAnthropic
from anthropic import AsyncAnthropicVertex
from anthropic import NOT_GIVEN
from anthropic import NotGiven
from anthropic import RateLimitError
from anthropic import types as anthropic_types
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from . import _prompt_cache
from ..utils import _json_utils
from ..utils._google_client_headers import get_tracking_headers
from ..utils._schema_utils import lowercase_schema_types
from .base_llm import BaseLlm
from .interactions_utils import extract_system_instruction
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from ..agents.context_cache_config import ContextCacheConfig
  from .llm_request import LlmRequest

__all__ = ["AnthropicLlm", "Claude", "AnthropicGenerateContentConfig"]

logger = logging.getLogger("google_adk." + __name__)

_ImageMediaType: TypeAlias = Literal[
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
]
_ANTHROPIC_IMAGE_MEDIA_TYPES = frozenset[str](get_args(_ImageMediaType))

_MessageBlockParam: TypeAlias = Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ThinkingBlockParam,
    anthropic_types.RedactedThinkingBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.DocumentBlockParam,
    anthropic_types.ToolUseBlockParam,
    anthropic_types.ToolResultBlockParam,
]

# The subset of block types Claude accepts inside a tool result.
_ToolResultContentBlockParam: TypeAlias = Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.DocumentBlockParam,
]

# Attributes an Anthropic client exposes once it has resolved a credential,
# whichever source it came from: a static API key, a static bearer token, or a
# credential provider discovered from the environment or from the on-disk
# Anthropic configuration. Only these three carry a credential - the client's
# own "could not resolve authentication method" error names the same three.
# `credentials` is absent on older supported SDK versions, so the lookup below
# tolerates a missing attribute.
_ANTHROPIC_CREDENTIAL_ATTRS = ("api_key", "auth_token", "credentials")

_RATE_LIMIT_POSSIBLE_FIX_MESSAGE = (
    "On how to mitigate this issue, please refer to:\n\n"
    "https://docs.anthropic.com/en/api/errors#http-errors"
)

# Claude rejects a cache breakpoint on a reasoning block.
_UNCACHEABLE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


# anthropic is an optional dependency, so mypy resolves the base class to Any.
class _AnthropicRateLimitError(RateLimitError):  # type: ignore[misc]
  """Represents a rate limit error received from Anthropic."""

  def __init__(self, rate_limit_error: RateLimitError):
    super().__init__(
        str(rate_limit_error),
        response=rate_limit_error.response,
        body=getattr(rate_limit_error, "body", None),
    )

  def __str__(self) -> str:
    base_message = super().__str__()
    return f"{_RATE_LIMIT_POSSIBLE_FIX_MESSAGE}\n\n{base_message}"


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


# Mapping of Anthropic stop_reason strings to FinishReason enum values
_STOP_REASON_MAPPING: dict[anthropic_types.StopReason, types.FinishReason] = {
    "end_turn": types.FinishReason.STOP,
    "stop_sequence": types.FinishReason.STOP,
    "tool_use": types.FinishReason.STOP,
    "pause_turn": types.FinishReason.STOP,
    "max_tokens": types.FinishReason.MAX_TOKENS,
    "refusal": types.FinishReason.SAFETY,
}


def to_google_genai_finish_reason(
    anthropic_stop_reason: Optional[anthropic_types.StopReason],
) -> types.FinishReason | None:
  """Maps Anthropic stop_reason to Google GenAI FinishReason."""
  if anthropic_stop_reason is None:
    return None
  return _STOP_REASON_MAPPING.get(
      anthropic_stop_reason, types.FinishReason.FINISH_REASON_UNSPECIFIED
  )


def _is_image_part(part: types.Part) -> bool:
  inline_data = part.inline_data
  return bool(
      inline_data is not None
      and inline_data.mime_type is not None
      and inline_data.mime_type.startswith("image/")
  )


def _is_pdf_part(part: types.Part) -> bool:
  inline_data = part.inline_data
  return bool(
      inline_data is not None
      and inline_data.mime_type is not None
      and inline_data.mime_type.split(";", 1)[0].strip() == "application/pdf"
  )


def _normalize_image_media_type(mime_type: str) -> _ImageMediaType:
  normalized = mime_type.split(";", 1)[0].strip().lower()
  if normalized not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
    raise ValueError(f"Unsupported Anthropic image MIME type: {mime_type}")
  return cast(_ImageMediaType, normalized)


def _function_response_media_blocks(
    function_response: types.FunctionResponse,
) -> list[_ToolResultContentBlockParam]:
  """Converts media a tool attached to its response into tool result blocks.

  Media Claude cannot carry in a tool result is dropped with a warning rather
  than raised on, because the tool that produced it is often third-party code
  the caller cannot change, and losing one image is better than losing the
  conversation.
  """
  blocks: list[_ToolResultContentBlockParam] = []
  for response_part in function_response.parts or []:
    blob = response_part.inline_data
    if blob is None or blob.data is None or not blob.mime_type:
      continue
    media_type = blob.mime_type.split(";", 1)[0].strip().lower()
    data = base64.b64encode(blob.data).decode()
    if media_type in _ANTHROPIC_IMAGE_MEDIA_TYPES:
      blocks.append(
          anthropic_types.ImageBlockParam(
              type="image",
              source=anthropic_types.Base64ImageSourceParam(
                  type="base64",
                  # Narrowed by the membership test above.
                  media_type=cast(_ImageMediaType, media_type),
                  data=data,
              ),
          )
      )
    elif media_type == "application/pdf":
      blocks.append(
          anthropic_types.DocumentBlockParam(
              type="document",
              source=anthropic_types.Base64PDFSourceParam(
                  type="base64",
                  media_type="application/pdf",
                  data=data,
              ),
          )
      )
    else:
      logger.warning(
          "Dropping tool result media of type %s, which Claude cannot receive"
          " in a tool result.",
          media_type,
      )
  return blocks


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
) -> _MessageBlockParam:
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
    function_call = part.function_call
    assert function_call.name
    tool_input: dict[str, object] = dict(function_call.args or {})

    return anthropic_types.ToolUseBlockParam(
        id=sanitizer.sanitize(function_call.id),
        name=function_call.name,
        input=tool_input,
        type="tool_use",
    )
  elif part.function_response:
    function_response = part.function_response
    content = ""
    response_data = function_response.response or {}

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

    # A tool can attach media alongside the serializable part of its result.
    # It travels in a dedicated field, so it has to be mapped over explicitly
    # or the model never sees it.
    media_blocks = _function_response_media_blocks(function_response)
    tool_result_content: Union[str, list[_ToolResultContentBlockParam]]
    if media_blocks:
      leading_text: list[_ToolResultContentBlockParam] = (
          [anthropic_types.TextBlockParam(type="text", text=content)]
          if content
          else []
      )
      tool_result_content = leading_text + media_blocks
    else:
      tool_result_content = content

    return anthropic_types.ToolResultBlockParam(
        tool_use_id=sanitizer.sanitize(function_response.id),
        type="tool_result",
        content=tool_result_content,
        is_error=False,
    )
  elif _is_image_part(part):
    inline_data = part.inline_data
    if (
        inline_data is None
        or inline_data.data is None
        or inline_data.mime_type is None
    ):
      raise ValueError("Anthropic image parts require MIME type and data")
    data = base64.b64encode(inline_data.data).decode()
    image_source = anthropic_types.Base64ImageSourceParam(
        type="base64",
        media_type=_normalize_image_media_type(inline_data.mime_type),
        data=data,
    )
    return anthropic_types.ImageBlockParam(
        type="image",
        source=image_source,
    )
  elif _is_pdf_part(part):
    inline_data = part.inline_data
    if inline_data is None or inline_data.data is None:
      raise ValueError("Anthropic PDF parts require data")
    data = base64.b64encode(inline_data.data).decode()
    pdf_source = anthropic_types.Base64PDFSourceParam(
        type="base64",
        media_type="application/pdf",
        data=data,
    )
    return anthropic_types.DocumentBlockParam(
        type="document",
        source=pdf_source,
    )
  elif part.executable_code:
    return anthropic_types.TextBlockParam(
        type="text",
        text="Code:```python\n" + (part.executable_code.code or "") + "\n```",
    )
  elif part.code_execution_result:
    return anthropic_types.TextBlockParam(
        text="Execution Result:```code_output\n"
        + (part.code_execution_result.output or "")
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
) -> _MessageBlockParam:
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
    function_call = part.function_call
    if function_call is None:
      raise ValueError("Function-call part factory returned no function call")
    function_call.id = content_block.id
    return part
  raise NotImplementedError(
      f"Unsupported content block type: {type(content_block)}"
  )


def _extract_cached_token_count(usage: Any) -> int | None:
  """Returns Anthropic cache-read tokens, the analog of cached_content tokens."""
  cached = getattr(usage, "cache_read_input_tokens", None)
  return cached if isinstance(cached, int) else None


def _extract_prompt_token_count(usage: anthropic_types.Usage) -> int:
  """Returns every input token billed for the turn.

  Anthropic reports tokens served from the prompt cache and tokens written to
  it in their own fields, disjoint from ``input_tokens``. The GenAI shape
  instead expects a single prompt count with the cached portion folded in --
  ``cached_content_token_count`` is a breakdown of it, not an addition to it.
  """
  total = 0
  for field in (
      "input_tokens",
      "cache_read_input_tokens",
      "cache_creation_input_tokens",
  ):
    value = getattr(usage, field, None)
    if isinstance(value, int):
      total += value
  return total


def _extract_thinking_token_count(
    usage: anthropic_types.Usage | anthropic_types.MessageDeltaUsage,
) -> int | None:
  """Returns Anthropic thinking tokens, the analog of thoughts tokens.

  Anthropic counts extended-thinking tokens inside ``output_tokens``, whereas
  the GenAI shape keeps the candidate and thought counts disjoint and sums them
  downstream. Callers therefore subtract this from ``output_tokens`` to get the
  candidate count; the value is clamped so that subtraction stays non-negative
  even if the two counters ever disagree.
  """
  details = getattr(usage, "output_tokens_details", None)
  thinking = getattr(details, "thinking_tokens", None)
  if not isinstance(thinking, int):
    return None
  output_tokens = getattr(usage, "output_tokens", None)
  if not isinstance(output_tokens, int):
    return thinking
  return min(thinking, output_tokens)


def _extract_cache_creation_token_count(usage: Any) -> int | None:
  """Returns Anthropic cache-write tokens, the analog of cache_creation tokens."""
  cached = getattr(usage, "cache_creation_input_tokens", None)
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

  prompt_tokens = _extract_prompt_token_count(message.usage)
  thinking_tokens = _extract_thinking_token_count(message.usage)
  usage_metadata = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=prompt_tokens,
      candidates_token_count=(
          message.usage.output_tokens - (thinking_tokens or 0)
      ),
      total_token_count=prompt_tokens + message.usage.output_tokens,
      cached_content_token_count=_extract_cached_token_count(message.usage),
      thoughts_token_count=thinking_tokens,
  )
  cache_creation = _extract_cache_creation_token_count(message.usage)
  if cache_creation is not None:
    object.__setattr__(
        usage_metadata, "cache_creation_input_tokens", cache_creation
    )

  return LlmResponse(
      content=types.Content(
          role="model",
          parts=parts,
      ),
      usage_metadata=usage_metadata,
      finish_reason=to_google_genai_finish_reason(message.stop_reason),
      model_version=message.model,
  )


def function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> anthropic_types.ToolParam:
  """Converts a function declaration to an Anthropic tool param."""
  assert function_declaration.name

  # Use parameters_json_schema if available, otherwise convert from parameters
  if function_declaration.parameters_json_schema:
    input_schema = copy.deepcopy(function_declaration.parameters_json_schema)
    lowercase_schema_types(input_schema)
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
    lowercase_schema_types(input_schema)

  return anthropic_types.ToolParam(
      name=function_declaration.name,
      description=function_declaration.description or "",
      input_schema=input_schema,
  )


def _to_cache_control(
    cache_config: ContextCacheConfig,
) -> anthropic_types.CacheControlEphemeralParam:
  """Maps the configured cache lifetime onto one Claude actually offers."""
  if _prompt_cache.use_one_hour_ttl(cache_config):
    return anthropic_types.CacheControlEphemeralParam(
        type="ephemeral", ttl="1h"
    )
  return anthropic_types.CacheControlEphemeralParam(type="ephemeral")


def _set_cache_control(
    block: anthropic_types.ToolUnionParam | _MessageBlockParam,
    cache_control: anthropic_types.CacheControlEphemeralParam,
) -> None:
  """Attaches a cache breakpoint to a tool definition or a content block.

  Every param the Anthropic SDK accepts is a plain dict at runtime; the unions
  of TypedDicts only describe their shape.
  """
  cast(dict[str, Any], block)["cache_control"] = cache_control


def _mark_last_cacheable_message_block(
    messages: list[anthropic_types.MessageParam],
    cache_control: anthropic_types.CacheControlEphemeralParam,
) -> None:
  """Puts a cache breakpoint at the end of the conversation so far.

  The search runs backwards because a turn can end in a reasoning block, which
  Claude refuses to cache, or carry no blocks at all once parts Claude cannot
  receive have been dropped.

  Args:
    messages: Conversation to mark, modified in place.
    cache_control: Breakpoint to attach.
  """
  for message in reversed(messages):
    content = message.get("content")
    if not isinstance(content, list):
      continue
    for block in reversed(content):
      block_type = cast(dict[str, Any], block).get("type")
      if block_type in _UNCACHEABLE_BLOCK_TYPES:
        continue
      _set_cache_control(block, cache_control)
      return


def _apply_cache_breakpoints(
    *,
    cache_config: ContextCacheConfig,
    system: str | NotGiven,
    messages: list[anthropic_types.MessageParam],
    tools: Iterable[anthropic_types.ToolUnionParam] | NotGiven,
) -> str | list[anthropic_types.TextBlockParam] | NotGiven:
  """Marks the reusable prefix of a request so Claude bills it as a cache hit.

  Claude charges the full input rate for the whole prompt on every turn unless
  a block carries a breakpoint. A breakpoint tells it to store the prefix
  ending at that block and to serve that prefix at the much lower cache-read
  rate on later turns.

  Claude reads the prompt as tools, then system, then messages, so a breakpoint
  on each of the three keeps the levels above a change still cached: editing
  the conversation leaves the tools and the system instruction cached, and
  editing the system instruction leaves the tools cached. Claude allows four
  breakpoints per request and these are three of them.

  The conversation breakpoint moves to the end of each request, and Claude
  finds the previous one by looking back at most twenty blocks. A turn that
  adds more blocks than that, such as one calling nine or more tools at once,
  therefore rewrites the conversation cache instead of reading it. The tools
  and system breakpoints are unaffected, so the stable head of the prompt is
  still served from the cache.

  Args:
    cache_config: Cache configuration for the request.
    system: System instruction to mark.
    messages: Conversation to mark, modified in place.
    tools: Tool definitions to mark, modified in place.

  Returns:
    The system instruction to send. Carrying a breakpoint turns it into a
    block list, so it is returned rather than modified in place.
  """
  cache_control = _to_cache_control(cache_config)

  if isinstance(tools, list) and tools:
    _set_cache_control(tools[-1], cache_control)

  _mark_last_cacheable_message_block(messages, cache_control)

  if isinstance(system, str):
    return [
        anthropic_types.TextBlockParam(
            type="text", text=system, cache_control=cache_control
        )
    ]
  return system


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

  client: Optional[Union[AsyncAnthropic, AsyncAnthropicVertex]] = Field(
      default=None, exclude=True
  )
  """An optional pre-configured Anthropic client."""

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r"claude-.*"]

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
    system: str | NotGiven = NOT_GIVEN
    if llm_request.config:
      system_str = extract_system_instruction(llm_request.config)
      if system_str:
        system = system_str

    system_param: str | list[anthropic_types.TextBlockParam] | NotGiven = system
    cache_config = _prompt_cache.resolve_cache_config(llm_request)
    if cache_config is not None:
      system_param = _apply_cache_breakpoints(
          cache_config=cache_config,
          system=system,
          messages=messages,
          tools=tools,
      )

    model_to_use = self._resolve_model_name(llm_request.model)
    kwargs: dict[str, Any] = {
        "model": model_to_use,
        "system": system_param,
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
    tools: Iterable[anthropic_types.ToolUnionParam] | NotGiven = NOT_GIVEN
    function_declarations: list[types.FunctionDeclaration] = []
    if llm_request.config and llm_request.config.tools:
      for configured_tool in llm_request.config.tools:
        if isinstance(configured_tool, types.Tool):
          function_declarations.extend(
              configured_tool.function_declarations or []
          )
    if function_declarations:
      tools = [
          function_declaration_to_tool_param(tool)
          for tool in function_declarations
      ]
    tool_choice = (
        anthropic_types.ToolChoiceAutoParam(type="auto")
        if llm_request.tools_dict
        else NOT_GIVEN
    )
    thinking = _build_anthropic_thinking_param(llm_request.config)

    try:
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
    except RateLimitError as rate_limit_error:
      raise _AnthropicRateLimitError(rate_limit_error) from rate_limit_error

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
    thinking_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_tokens: int | None = None
    stop_reason: Optional[anthropic_types.StopReason] = None
    model_version: Optional[str] = None

    async for event in raw_stream:
      if event.type == "message_start":
        input_tokens = _extract_prompt_token_count(event.message.usage)
        output_tokens = event.message.usage.output_tokens
        thinking_tokens = _extract_thinking_token_count(event.message.usage)
        cached_input_tokens = _extract_cached_token_count(event.message.usage)
        cache_creation_tokens = _extract_cache_creation_token_count(
            event.message.usage
        )
        model_version = event.message.model

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
              model_version=model_version,
              partial=True,
          )
        elif isinstance(delta, anthropic_types.SignatureDelta):
          # Claude streams the thinking block's cryptographic signature as a
          # separate delta near the end of the block. Accumulate it so the
          # aggregated thinking Part below carries ``thought_signature``.
          # Without it the reasoning block cannot round-trip back to Claude on
          # the next request -- extended thinking + tool use requires echoing
          # the signed thinking blocks, and re-serializing history for the
          # follow-up call would otherwise fail. Not surfaced as a partial (the
          # signature is opaque, not user-visible text).
          thinking_blocks.setdefault(
              event.index,
              _ThinkingAccumulator(thinking="", signature=""),
          )
          thinking_blocks[event.index].signature += delta.signature
        elif isinstance(delta, anthropic_types.TextDelta):
          text_blocks.setdefault(event.index, "")
          text_blocks[event.index] += delta.text
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text=delta.text)],
              ),
              model_version=model_version,
              partial=True,
          )
        elif isinstance(delta, anthropic_types.InputJSONDelta):
          if event.index in tool_use_blocks:
            tool_use_blocks[event.index].args_json += delta.partial_json

      elif event.type == "message_delta":
        # ``message_delta`` carries the authoritative cumulative counts, so the
        # thinking detail is refreshed alongside the total it is nested in.
        output_tokens = event.usage.output_tokens
        thinking_tokens = _extract_thinking_token_count(event.usage)
        if event.delta and event.delta.stop_reason:
          stop_reason = event.delta.stop_reason

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
        thinking_acc = thinking_blocks[idx]
        part = types.Part(text=thinking_acc.thinking, thought=True)
        if thinking_acc.signature:
          part.thought_signature = thinking_acc.signature.encode("utf-8")
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
        tool_acc = tool_use_blocks[idx]
        args = (
            _json_utils.safe_json_loads(tool_acc.args_json)
            if tool_acc.args_json
            else {}
        )
        part = types.Part.from_function_call(name=tool_acc.name, args=args)
        function_call = part.function_call
        if function_call is None:
          raise ValueError(
              "Function-call part factory returned no function call"
          )
        function_call.id = tool_acc.id
        all_parts.append(part)

    usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=input_tokens,
        candidates_token_count=output_tokens - (thinking_tokens or 0),
        total_token_count=input_tokens + output_tokens,
        cached_content_token_count=cached_input_tokens,
        thoughts_token_count=thinking_tokens,
    )
    if cache_creation_tokens is not None:
      object.__setattr__(
          usage_metadata, "cache_creation_input_tokens", cache_creation_tokens
      )

    yield LlmResponse(
        content=types.Content(role="model", parts=all_parts),
        usage_metadata=usage_metadata,
        finish_reason=to_google_genai_finish_reason(stop_reason),
        model_version=model_version,
        partial=False,
    )

  @cached_property
  def _anthropic_client(self) -> AsyncAnthropic | AsyncAnthropicVertex:
    if self.client:
      return self.client
    client = AsyncAnthropic()
    # Let the SDK run its own credential resolution first, then ask the client
    # what it found. Enumerating credential sources here would reject setups
    # the SDK handles perfectly well, such as a signed-in on-disk profile with
    # no credential environment variable set at all.
    if not any(
        getattr(client, attr, None) for attr in _ANTHROPIC_CREDENTIAL_ATTRS
    ):
      raise ValueError(
          "No Anthropic credential was found for calling Claude through the"
          " Anthropic API. Set ANTHROPIC_API_KEY to a key from the Anthropic"
          " Console, e.g. `export ANTHROPIC_API_KEY=<your-key>`, or configure"
          " any other credential the Anthropic SDK can discover."
      )
    return client


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
    if self.client is not None:
      if not isinstance(self.client, AsyncAnthropicVertex):
        raise ValueError("Claude requires an AsyncAnthropicVertex client.")
      return self.client
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
          f"Model {self.model!r} resolves to Claude served from Vertex AI, so"
          " GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set to the"
          " project and region serving the model. To call the Anthropic API"
          " directly with an ANTHROPIC_API_KEY instead, pass a model instance"
          " configured for the Anthropic API rather than a bare model name."
      )

    return AsyncAnthropicVertex(
        project_id=project_id,
        region=location,
        default_headers=get_tracking_headers(),
    )
