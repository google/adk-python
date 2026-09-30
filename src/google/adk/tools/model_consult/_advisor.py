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

"""Advisor LLM invocation for the `model_consult` tool.

Calls a `BaseLlm` directly via `generate_content_async(req, stream=False)`
with `config.tools = []` and `config.tool_config = None` so the advisor returns
text guidance only and cannot call tools or enter an agent loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Sequence
import copy
from dataclasses import dataclass
import logging
import time

from google.genai import types

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...models.registry import LLMRegistry
from ...telemetry import _metrics
from ...telemetry import tracing
from ...telemetry._token_usage import TokenUsage
from ...utils.model_name_utils import is_gemini_model

logger = logging.getLogger('google_adk.' + __name__)

_THINKING_LEVEL_MAP: dict[str, types.ThinkingLevel] = {
    'minimal': types.ThinkingLevel.MINIMAL,
    'low': types.ThinkingLevel.LOW,
    'medium': types.ThinkingLevel.MEDIUM,
    'high': types.ThinkingLevel.HIGH,
}


class AdvisorError(RuntimeError):  # pylint: disable=g-bad-exception-name
  """Raised when the advisor model call fails or returns unusable output."""


@dataclass(frozen=True, kw_only=True)
class AdvisorUsage:
  """Token accounting for a single advisor call (or cumulative across calls).

  Attributes:
    prompt_tokens: Input tokens billed for the prompt (including tool-use prompt
      tokens when reported).
    output_tokens: Candidate output tokens (excluding thoughts).
    thoughts_tokens: Reasoning/thinking tokens consumed by the advisor.
    cached_tokens: Prompt tokens served from a context cache.
    total_tokens: Total tokens consumed (`prompt + output + thoughts` when not
      explicitly reported by the provider).
  """

  prompt_tokens: int = 0
  output_tokens: int = 0
  thoughts_tokens: int = 0
  cached_tokens: int = 0
  total_tokens: int = 0

  @classmethod
  def from_metadata(
      cls, meta: types.GenerateContentResponseUsageMetadata | None
  ) -> AdvisorUsage:
    """Builds an `AdvisorUsage` snapshot from GenAI usage metadata.

    Args:
      meta: Usage metadata from `LlmResponse.usage_metadata`, or `None`.

    Returns:
      An `AdvisorUsage` populated from `meta`, or all-zero counts if `None`.
    """
    if meta is None:
      return cls()
    buckets = TokenUsage.from_usage_metadata(meta)
    prompt = max(0, buckets.input_tokens or 0)
    output = max(0, buckets.candidate_output_tokens or 0)
    thoughts = max(0, buckets.reasoning_output_tokens or 0)
    cached = max(0, buckets.cache_read_input_tokens or 0)
    raw_total = max(0, meta.total_token_count or 0)
    total = raw_total or (prompt + output + thoughts)
    return cls(
        prompt_tokens=prompt,
        output_tokens=output,
        thoughts_tokens=thoughts,
        cached_tokens=cached,
        total_tokens=total,
    )

  def __add__(self, other: AdvisorUsage) -> AdvisorUsage:
    if not isinstance(other, AdvisorUsage):
      return NotImplemented
    return AdvisorUsage(
        prompt_tokens=self.prompt_tokens + other.prompt_tokens,
        output_tokens=self.output_tokens + other.output_tokens,
        thoughts_tokens=self.thoughts_tokens + other.thoughts_tokens,
        cached_tokens=self.cached_tokens + other.cached_tokens,
        total_tokens=self.total_tokens + other.total_tokens,
    )

  def to_dict(self) -> dict[str, int]:
    """Returns token counts as a JSON-serializable dictionary."""
    return {
        'prompt_tokens': self.prompt_tokens,
        'output_tokens': self.output_tokens,
        'thoughts_tokens': self.thoughts_tokens,
        'cached_tokens': self.cached_tokens,
        'total_tokens': self.total_tokens,
    }


@dataclass(frozen=True, kw_only=True)
class AdvisorResult:
  """Outcome of a single advisor model consultation.

  Attributes:
    text: Visible guidance text produced by the advisor model.
    model: Configured model identifier on the advisor `BaseLlm`.
    model_version: Provider-reported model version string, if available.
    usage: Token usage snapshot for the consultation.
    latency_ms: End-to-end wall-clock duration of the consultation in ms.
  """

  text: str
  model: str
  model_version: str | None
  usage: AdvisorUsage
  latency_ms: float


def resolve_thinking_level(
    level: str | types.ThinkingLevel | None,
) -> types.ThinkingLevel | None:
  """Maps a user-supplied thinking level to `types.ThinkingLevel`.

  Args:
    level: One of `'minimal'`, `'low'`, `'medium'`, `'high'`
      (case-insensitive), `'none'` / `'off'` / `''` / `None` to leave thinking
      unset, or a `types.ThinkingLevel` enum value.

  Returns:
    The corresponding `types.ThinkingLevel`, or `None` if disabled.

  Raises:
    ValueError: If `level` is not a recognized thinking level.
  """
  if level is None:
    return None
  if isinstance(level, types.ThinkingLevel):
    if level == types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED:
      return None
    return level
  if not isinstance(level, str):
    raise ValueError(
        f'Invalid advisor thinking_level {level!r}; expected a string or '
        'types.ThinkingLevel.'
    )
  key = level.strip().lower()
  if key in ('', 'none', 'off'):
    return None
  if key not in _THINKING_LEVEL_MAP:
    valid = sorted([*_THINKING_LEVEL_MAP.keys(), 'off'])
    raise ValueError(
        f'Invalid advisor thinking_level {level!r}; expected one of {valid}.'
    )
  return _THINKING_LEVEL_MAP[key]


def resolve_advisor_llm(model: str | BaseLlm) -> BaseLlm:
  """Resolves a model name or `BaseLlm` instance into a `BaseLlm`.

  Args:
    model: Either an already-constructed `BaseLlm` or a model identifier
      accepted by `LLMRegistry.new_llm` (for example `'gemini-2.5-pro'`).

  Returns:
    A `BaseLlm` instance for the advisor model.

  Raises:
    ValueError: If `model` is neither a `BaseLlm` nor a non-empty string.
  """
  if isinstance(model, BaseLlm):
    return model
  if isinstance(model, str) and model.strip():
    return LLMRegistry.new_llm(model.strip())
  raise ValueError(
      f'Invalid advisor_model {model!r}; expected a non-empty model string or '
      'a BaseLlm instance.'
  )


def _build_request(
    *,
    llm: BaseLlm,
    contents: Sequence[types.Content],
    system_instruction: str,
    thinking_level: types.ThinkingLevel | None,
    max_output_tokens: int | None,
    base_config: types.GenerateContentConfig | None,
    clear_thinking_config: bool = False,
) -> LlmRequest:
  """Constructs a tool-free `LlmRequest` for the advisor call."""
  config = (
      copy.deepcopy(base_config)
      if base_config is not None
      else types.GenerateContentConfig()
  )
  config.system_instruction = system_instruction
  config.tools = []
  config.tool_config = None
  if max_output_tokens is not None:
    config.max_output_tokens = max_output_tokens
  if clear_thinking_config:
    config.thinking_config = None
  elif thinking_level is not None:
    existing = config.thinking_config
    config.thinking_config = types.ThinkingConfig(
        thinking_level=thinking_level,
        include_thoughts=existing.include_thoughts if existing else None,
    )
  return LlmRequest(
      model=llm.model,
      contents=list(contents),
      config=config,
  )


async def _collect(
    llm: BaseLlm,
    response_gen: AsyncGenerator[LlmResponse, None],
    responses: list[LlmResponse],
) -> tuple[
    str,
    str | None,
    types.FinishReason | None,
    AdvisorUsage,
]:
  """Iterates `response_gen` and extracts visible text and metadata."""
  text_chunks: list[str] = []
  model_version: str | None = None
  finish_reason: types.FinishReason | None = None
  last_usage_meta: types.GenerateContentResponseUsageMetadata | None = None

  try:
    async for response in response_gen:
      responses.append(response)
      if response.model_version:
        model_version = response.model_version
      if response.finish_reason:
        finish_reason = response.finish_reason
      elif _hit_output_cap(response.error_code):
        finish_reason = types.FinishReason.MAX_TOKENS
      if response.usage_metadata is not None:
        # Take the last reading rather than summing across yields: adapters
        # report cumulative token counts on the final non-partial response.
        last_usage_meta = response.usage_metadata
      if (
          response.error_code
          and not _hit_output_cap(response.error_code)
          and not _hit_output_cap(response.finish_reason)
      ):
        raise AdvisorError(
            f'Advisor ({llm.model}) returned error {response.error_code}: '
            f'{response.error_message or "no message"}'
        )
      if response.partial:
        continue
      if response.content and response.content.parts:
        for part in response.content.parts:
          if getattr(part, 'thought', False):
            continue
          if part.text:
            text_chunks.append(part.text)
  finally:
    await response_gen.aclose()

  text = ''.join(text_chunks).strip()
  usage = AdvisorUsage.from_metadata(last_usage_meta)
  return text, model_version, finish_reason, usage


def _record_telemetry(
    *,
    agent_name: str,
    elapsed_s: float,
    request: LlmRequest,
    responses: Sequence[LlmResponse],
    error: Exception | None = None,
) -> None:
  """Emits standard ADK OpenTelemetry client duration and token metrics."""
  try:
    # pylint: disable=protected-access
    if (
        tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
        and is_gemini_model(request.model)
    ):
      return

    normalized_responses: list[LlmResponse] = []
    last_usage_meta: types.GenerateContentResponseUsageMetadata | None = None
    last_model_version: str | None = None
    for resp in responses:
      if resp.model_version:
        last_model_version = resp.model_version
      if resp.usage_metadata is not None:
        last_usage_meta = resp.usage_metadata
    if responses:
      tail = responses[-1].model_copy(
          update={
              'model_version': (
                  last_model_version or responses[-1].model_version
              ),
              'usage_metadata': (
                  last_usage_meta
                  if last_usage_meta is not None
                  else responses[-1].usage_metadata
              ),
          }
      )
      normalized_responses = [*responses[:-1], tail]

    _metrics.record_client_operation_duration(
        agent_name=agent_name,
        elapsed_s=elapsed_s,
        llm_request=request,
        responses=normalized_responses,
        error=error,
    )
    if last_usage_meta is not None and normalized_responses:
      _metrics.record_client_token_usage(
          agent_name=agent_name,
          llm_request=request,
          responses=normalized_responses,
      )
  except Exception:  # pylint: disable=broad-exception-caught
    logger.debug(
        'Failed to record telemetry for advisor call (%s).',
        request.model,
        exc_info=True,
    )


async def call_advisor(
    llm: BaseLlm,
    contents: Sequence[types.Content],
    *,
    system_instruction: str,
    thinking_level: types.ThinkingLevel | None = None,
    max_output_tokens: int | None = None,
    timeout_seconds: float | None = None,
    generate_content_config: types.GenerateContentConfig | None = None,
    agent_name: str = 'model_consult',
) -> AdvisorResult:
  """Executes one non-streaming advisor call and returns its guidance text.

  If `thinking_level` (or `generate_content_config.thinking_config`) is set
  and the underlying model rejects `thinking_config` (for example a model or
  third-party adapter that does not support thinking levels), the call retries
  once without `thinking_config`.

  Args:
    llm: The resolved advisor `BaseLlm` instance.
    contents: Handover conversation contents from `build_advisor_contents`.
    system_instruction: System prompt instructing the advisor how to respond.
    thinking_level: Optional `types.ThinkingLevel` for the advisor call.
    max_output_tokens: Optional cap on total generated tokens (thoughts + text).
    timeout_seconds: Optional positive wall-clock timeout in seconds.
    generate_content_config: Optional base `GenerateContentConfig` to copy.
    agent_name: Agent attribute recorded on OpenTelemetry client metrics.

  Returns:
    An `AdvisorResult` with the advisor's visible response text, model info,
    token usage, and wall-clock latency in milliseconds.

  Raises:
    ValueError: If `timeout_seconds` is less than or equal to zero.
    AdvisorError: If the call times out, errors, or produces no visible text.
  """
  if timeout_seconds is not None and timeout_seconds <= 0:
    raise ValueError(
        f'timeout_seconds must be positive; got {timeout_seconds!r}.'
    )

  req = _build_request(
      llm=llm,
      contents=contents,
      system_instruction=system_instruction,
      thinking_level=thinking_level,
      max_output_tokens=max_output_tokens,
      base_config=generate_content_config,
  )
  can_retry_without_thinking = req.config.thinking_config is not None
  call_t0 = time.perf_counter()
  deadline = call_t0 + timeout_seconds if timeout_seconds is not None else None

  while True:
    attempt_t0 = time.perf_counter()
    responses: list[LlmResponse] = []
    try:
      coro = _collect(
          llm,
          llm.generate_content_async(req, stream=False),
          responses,
      )
      if deadline is not None:
        remaining_timeout = max(0.0, deadline - time.perf_counter())
        text, model_version, finish_reason, usage = await asyncio.wait_for(
            coro, timeout=remaining_timeout
        )
      else:
        text, model_version, finish_reason, usage = await coro

      effective_max_tokens = req.config.max_output_tokens
      if not text and _hit_output_cap(finish_reason):
        raise AdvisorError(
            f'Advisor ({llm.model}) produced no visible text before hitting '
            f'max_output_tokens={effective_max_tokens} (thoughts consumed '
            f'{usage.thoughts_tokens} tokens). Increase max_output_tokens or '
            'lower thinking_level.'
        )

      if not text:
        raise AdvisorError(
            f'Advisor ({llm.model}) returned an empty response '
            f'(finish_reason={finish_reason}).'
        )
      break
    # Before Python 3.11, asyncio.TimeoutError is not the builtin
    # TimeoutError, so catch both to cover asyncio and transport timeouts.
    except (asyncio.TimeoutError, TimeoutError) as exc:
      _record_telemetry(
          agent_name=agent_name,
          elapsed_s=time.perf_counter() - attempt_t0,
          request=req,
          responses=responses,
          error=exc,
      )
      if timeout_seconds is not None:
        raise AdvisorError(
            f'Advisor ({llm.model}) timed out after {timeout_seconds}s.'
        ) from exc
      raise AdvisorError(f'Advisor ({llm.model}) timed out: {exc}') from exc
    except AdvisorError as exc:
      _record_telemetry(
          agent_name=agent_name,
          elapsed_s=time.perf_counter() - attempt_t0,
          request=req,
          responses=responses,
          error=exc,
      )
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _record_telemetry(
          agent_name=agent_name,
          elapsed_s=time.perf_counter() - attempt_t0,
          request=req,
          responses=responses,
          error=exc,
      )
      if can_retry_without_thinking and _is_thinking_config_error(exc):
        can_retry_without_thinking = False
        logger.info(
            'Advisor model %s rejected thinking_config (%s); retrying without '
            'thinking_config.',
            llm.model,
            exc,
        )
        req = _build_request(
            llm=llm,
            contents=contents,
            system_instruction=system_instruction,
            thinking_level=None,
            max_output_tokens=max_output_tokens,
            base_config=generate_content_config,
            clear_thinking_config=True,
        )
        continue
      raise AdvisorError(f'Advisor ({llm.model}) call failed: {exc}') from exc

  _record_telemetry(
      agent_name=agent_name,
      elapsed_s=time.perf_counter() - attempt_t0,
      request=req,
      responses=responses,
  )
  latency_ms = (time.perf_counter() - call_t0) * 1000.0

  if _hit_output_cap(finish_reason):
    text = f'{text}\n\n[advisor guidance truncated at max_output_tokens]'

  return AdvisorResult(
      text=text,
      model=llm.model,
      model_version=model_version,
      usage=usage,
      latency_ms=latency_ms,
  )


def _hit_output_cap(
    finish_reason: types.FinishReason | str | None,
) -> bool:
  """Returns True if generation stopped because `MAX_TOKENS` was reached."""
  if finish_reason is None:
    return False
  return str(finish_reason).upper().endswith('MAX_TOKENS')


def _is_thinking_config_error(exc: BaseException) -> bool:
  """Heuristic check for errors caused by an unsupported `thinking_config`."""
  msg = str(exc).lower()
  if any(
      field in msg
      for field in (
          'thinking_config',
          'thinkingconfig',
          'thinking_level',
          'thinking level',
          'thinking_budget',
          'thinking budget',
      )
  ):
    return True
  return 'thinking' in msg and any(
      token in msg
      for token in (
          'unsupported',
          'not support',
          'unknown',
          'unexpected',
          'only available',
          'not allowed',
          'cannot',
      )
  )
