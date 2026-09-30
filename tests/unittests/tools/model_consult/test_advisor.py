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

"""Unit tests for `google.adk.tools.model_consult._advisor`."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest import mock

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.telemetry import _metrics
from google.adk.telemetry import tracing
from google.adk.tools.model_consult._advisor import AdvisorError
from google.adk.tools.model_consult._advisor import AdvisorUsage
from google.adk.tools.model_consult._advisor import call_advisor
from google.adk.tools.model_consult._advisor import resolve_advisor_llm
from google.adk.tools.model_consult._advisor import resolve_thinking_level
from google.genai import types
from pydantic import Field
import pytest


class _FakeAdvisorLlm(BaseLlm):
  """Test double for `BaseLlm` yielding scripted responses."""

  model: str = 'fake-advisor-pro'
  scripted_outcomes: list[list[LlmResponse] | BaseException] = Field(
      default_factory=list
  )
  recorded_requests: list[LlmRequest] = Field(default_factory=list)
  recorded_streams: list[bool] = Field(default_factory=list)
  delay_seconds: float = 0.0

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.recorded_requests.append(llm_request.model_copy(deep=True))
    self.recorded_streams.append(stream)
    if self.delay_seconds > 0:
      await asyncio.sleep(self.delay_seconds)
    if not self.scripted_outcomes:
      return
    outcome = self.scripted_outcomes.pop(0)
    if isinstance(outcome, BaseException):
      raise outcome
    for resp in outcome:
      yield resp


def _sample_contents() -> tuple[types.Content, ...]:
  return (
      types.Content(
          role='user',
          parts=[
              types.Part.from_text(text='How should I structure this retry?')
          ],
      ),
  )


@pytest.mark.parametrize(
    ('raw_level', 'expected'),
    [
        (None, None),
        ('', None),
        ('   ', None),
        ('none', None),
        ('OFF', None),
        ('minimal', types.ThinkingLevel.MINIMAL),
        ('LOW', types.ThinkingLevel.LOW),
        ('  Medium  ', types.ThinkingLevel.MEDIUM),
        ('high', types.ThinkingLevel.HIGH),
        (types.ThinkingLevel.HIGH, types.ThinkingLevel.HIGH),
        (types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED, None),
    ],
)
def test_resolve_thinking_level_valid(
    raw_level: str | types.ThinkingLevel | None,
    expected: types.ThinkingLevel | None,
):
  """Normalizes valid thinking level strings, enums, and off/none values."""
  assert resolve_thinking_level(raw_level) == expected


@pytest.mark.parametrize('bad_level', ['ultra', 'maximum', 42])
def test_resolve_thinking_level_invalid_raises(bad_level):
  """Raises ValueError when given an unrecognized thinking level."""
  with pytest.raises(ValueError, match='Invalid advisor thinking_level'):
    resolve_thinking_level(bad_level)


def test_resolve_advisor_llm_passes_through_instance():
  """Returns an already-constructed BaseLlm instance unchanged."""
  llm = _FakeAdvisorLlm()
  assert resolve_advisor_llm(llm) is llm


def test_resolve_advisor_llm_resolves_string_via_registry():
  """Strips and resolves a model string via LLMRegistry.new_llm."""
  fake_llm = _FakeAdvisorLlm()
  with mock.patch.object(
      LLMRegistry, 'new_llm', autospec=True, return_value=fake_llm
  ) as mock_new_llm:
    resolved = resolve_advisor_llm('  gemini-2.5-pro  ')
  assert resolved is fake_llm
  mock_new_llm.assert_called_once_with('gemini-2.5-pro')


@pytest.mark.parametrize('bad_model', ['', '   ', None])
def test_resolve_advisor_llm_invalid_raises(bad_model):
  """Raises ValueError when advisor_model is empty or not a string/BaseLlm."""
  with pytest.raises(ValueError, match='Invalid advisor_model'):
    resolve_advisor_llm(bad_model)  # type: ignore[arg-type]


def test_advisor_usage_from_metadata_and_addition():
  """Computes token totals, clamps negative sentinels, and adds snapshots."""
  assert AdvisorUsage.from_metadata(None) == AdvisorUsage()

  meta_fallback_total = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=100,
      tool_use_prompt_token_count=15,
      candidates_token_count=40,
      thoughts_token_count=60,
      cached_content_token_count=25,
      total_token_count=None,
  )
  u1 = AdvisorUsage.from_metadata(meta_fallback_total)
  assert u1 == AdvisorUsage(
      prompt_tokens=115,
      output_tokens=40,
      thoughts_tokens=60,
      cached_tokens=25,
      total_tokens=215,
  )

  meta_explicit_total = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=10,
      candidates_token_count=5,
      thoughts_token_count=2,
      cached_content_token_count=-1,
      total_token_count=50,
  )
  u2 = AdvisorUsage.from_metadata(meta_explicit_total)
  assert u2 == AdvisorUsage(
      prompt_tokens=10,
      output_tokens=5,
      thoughts_tokens=2,
      cached_tokens=0,
      total_tokens=50,
  )

  combined = u1 + u2
  assert combined.to_dict() == {
      'prompt_tokens': 125,
      'output_tokens': 45,
      'thoughts_tokens': 62,
      'cached_tokens': 25,
      'total_tokens': 265,
  }
  with pytest.raises(TypeError):
    _ = u1 + 'invalid'  # type: ignore[operator]


@pytest.mark.asyncio
async def test_call_advisor_happy_path_filters_thoughts_and_partials():
  """Collects visible text across chunks and records OTel metrics."""
  base_cfg = types.GenerateContentConfig(
      temperature=0.2,
      max_output_tokens=1024,
      tool_config=types.ToolConfig(
          function_calling_config=types.FunctionCallingConfig(
              mode=types.FunctionCallingConfigMode.ANY
          )
      ),
      thinking_config=types.ThinkingConfig(include_thoughts=True),
  )
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[[
          LlmResponse(
              partial=True,
              content=types.Content(
                  role='model',
                  parts=[types.Part.from_text(text='partial duplicate')],
              ),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=999,
                  candidates_token_count=5,
                  total_token_count=1004,
              ),
          ),
          LlmResponse(
              partial=False,
              model_version='gemini-2.5-pro-001',
              finish_reason=types.FinishReason.STOP,
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(text='internal thought', thought=True),
                      types.Part.from_text(text='  Use exponential '),
                  ],
              ),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=50,
                  candidates_token_count=20,
                  thoughts_token_count=30,
                  total_token_count=100,
              ),
          ),
          LlmResponse(
              partial=False,
              model_version=None,
              finish_reason=None,
              content=types.Content(
                  role='model',
                  parts=[types.Part.from_text(text='backoff.  ')],
              ),
              usage_metadata=None,
          ),
      ]]
  )

  with (
      mock.patch.object(
          _metrics, 'record_client_operation_duration', autospec=True
      ) as mock_duration,
      mock.patch.object(
          _metrics, 'record_client_token_usage', autospec=True
      ) as mock_tokens,
  ):
    result = await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Give concise advice.',
        thinking_level=types.ThinkingLevel.HIGH,
        generate_content_config=base_cfg,
    )

  assert result.text == 'Use exponential backoff.'
  assert result.model == 'fake-advisor-pro'
  assert result.model_version == 'gemini-2.5-pro-001'
  assert result.usage == AdvisorUsage(
      prompt_tokens=50,
      output_tokens=20,
      thoughts_tokens=30,
      cached_tokens=0,
      total_tokens=100,
  )
  assert result.latency_ms > 0.0

  assert llm.recorded_streams == [False]
  assert len(llm.recorded_requests) == 1
  sent_cfg = llm.recorded_requests[0].config
  assert sent_cfg.system_instruction == 'Give concise advice.'
  assert sent_cfg.tools == []
  assert sent_cfg.tool_config is None
  assert sent_cfg.max_output_tokens == 1024
  assert sent_cfg.temperature == 0.2
  assert sent_cfg.thinking_config.thinking_level == types.ThinkingLevel.HIGH
  assert sent_cfg.thinking_config.include_thoughts is True
  assert base_cfg.thinking_config.thinking_level is None

  mock_duration.assert_called_once()
  assert mock_duration.call_args.kwargs['agent_name'] == 'model_consult'
  assert mock_duration.call_args.kwargs['error'] is None
  assert (
      mock_duration.call_args.kwargs['responses'][-1].model_version
      == 'gemini-2.5-pro-001'
  )
  mock_tokens.assert_called_once()
  assert mock_tokens.call_args.kwargs['agent_name'] == 'model_consult'
  assert (
      mock_tokens.call_args.kwargs['responses'][
          -1
      ].usage_metadata.total_token_count
      == 100
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'err_msg',
    [
        'thinking_config is unsupported for this model',
        'thinking_level is not supported',
        'Model claude-3-5-haiku does not support thinking',
        (
            'thinking_budget must be set explicitly when ThinkingConfig is '
            'provided for Anthropic models'
        ),
        'Thinking is only available on Gemini 2.5 and newer models',
    ],
)
async def test_call_advisor_retries_without_thinking_config_on_rejection(
    err_msg: str,
):
  """Retries once without thinking_config and records telemetry for both."""
  base_cfg = types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
          thinking_level=types.ThinkingLevel.HIGH
      )
  )
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[
          ValueError(err_msg),
          [
              LlmResponse(
                  finish_reason=types.FinishReason.STOP,
                  content=types.Content(
                      role='model',
                      parts=[types.Part.from_text(text='Fallback succeeded.')],
                  ),
              )
          ],
      ]
  )

  with mock.patch.object(
      _metrics, 'record_client_operation_duration', autospec=True
  ) as mock_duration:
    result = await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
        thinking_level=None,
        generate_content_config=base_cfg,
    )

  assert result.text == 'Fallback succeeded.'
  assert len(llm.recorded_requests) == 2
  assert llm.recorded_requests[0].config.thinking_config is not None
  assert llm.recorded_requests[1].config.thinking_config is None
  assert mock_duration.call_count == 2
  assert isinstance(mock_duration.call_args_list[0].kwargs['error'], ValueError)
  assert mock_duration.call_args_list[1].kwargs['error'] is None


@pytest.mark.asyncio
async def test_call_advisor_preserves_or_overrides_caller_thinking_budget():
  """Preserves thinking_budget when thinking_level=None; overrides when set."""
  base_cfg = types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
          thinking_budget=2048, include_thoughts=True
      )
  )
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[
          [
              LlmResponse(
                  finish_reason=types.FinishReason.STOP,
                  content=types.Content(
                      role='model',
                      parts=[types.Part.from_text(text='Used budget.')],
                  ),
              )
          ],
          [
              LlmResponse(
                  finish_reason=types.FinishReason.STOP,
                  content=types.Content(
                      role='model',
                      parts=[types.Part.from_text(text='Used level.')],
                  ),
              )
          ],
      ]
  )

  result_preserved = await call_advisor(
      llm,
      _sample_contents(),
      system_instruction='Advisor system prompt.',
      thinking_level=None,
      generate_content_config=base_cfg,
  )
  assert result_preserved.text == 'Used budget.'
  sent_preserved = llm.recorded_requests[0].config.thinking_config
  assert sent_preserved.thinking_budget == 2048
  assert sent_preserved.thinking_level is None

  result_overridden = await call_advisor(
      llm,
      _sample_contents(),
      system_instruction='Advisor system prompt.',
      thinking_level=types.ThinkingLevel.HIGH,
      generate_content_config=base_cfg,
  )
  assert result_overridden.text == 'Used level.'
  sent_overridden = llm.recorded_requests[1].config.thinking_config
  assert sent_overridden.thinking_level == types.ThinkingLevel.HIGH
  assert sent_overridden.thinking_budget is None
  assert sent_overridden.include_thoughts is True


@pytest.mark.asyncio
async def test_call_advisor_does_not_retry_unrelated_invalid_argument_errors():
  """Does not retry 400 INVALID_ARGUMENT errors unrelated to thinking config."""
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[
          RuntimeError(
              '400 INVALID_ARGUMENT: Invalid value at contents[0] '
              '(text: "I am thinking about this")'
          ),
          [
              LlmResponse(
                  content=types.Content(
                      role='model',
                      parts=[types.Part.from_text(text='Should not run')],
                  )
              )
          ],
      ]
  )

  with pytest.raises(AdvisorError, match='400 INVALID_ARGUMENT'):
    await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
        thinking_level=types.ThinkingLevel.HIGH,
    )
  assert len(llm.recorded_requests) == 1


@pytest.mark.asyncio
async def test_call_advisor_max_tokens_with_no_visible_text_raises():
  """Raises thought-starvation AdvisorError even with error_code=MAX_TOKENS."""
  base_cfg = types.GenerateContentConfig(max_output_tokens=512)
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[[
          LlmResponse(
              finish_reason=types.FinishReason.MAX_TOKENS,
              error_code=types.FinishReason.MAX_TOKENS,
              content=types.Content(role='model', parts=[]),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=100,
                  thoughts_token_count=512,
                  total_token_count=612,
              ),
          )
      ]]
  )

  with mock.patch.object(
      _metrics, 'record_client_operation_duration', autospec=True
  ) as mock_duration:
    with pytest.raises(
        AdvisorError,
        match=(
            r'no visible text before hitting max_output_tokens=512.*512 tokens'
        ),
    ):
      await call_advisor(
          llm,
          _sample_contents(),
          system_instruction='Advisor system prompt.',
          generate_content_config=base_cfg,
      )
  mock_duration.assert_called_once()
  assert isinstance(mock_duration.call_args.kwargs['error'], AdvisorError)


@pytest.mark.asyncio
async def test_call_advisor_max_tokens_with_partial_text_and_error_code():
  """Returns truncated text when LiteLlm sets error_code=MAX_TOKENS."""
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[[
          LlmResponse(
              finish_reason=types.FinishReason.MAX_TOKENS,
              error_code=types.FinishReason.MAX_TOKENS,
              error_message='Maximum tokens reached',
              content=types.Content(
                  role='model',
                  parts=[types.Part.from_text(text='Step 1: check logs.')],
              ),
          )
      ]]
  )

  result = await call_advisor(
      llm,
      _sample_contents(),
      system_instruction='Advisor system prompt.',
      max_output_tokens=64,
  )
  assert result.text == (
      'Step 1: check logs.\n\n[advisor guidance truncated at max_output_tokens]'
  )


@pytest.mark.asyncio
async def test_call_advisor_empty_response_on_stop_raises():
  """Raises AdvisorError when finish_reason is STOP/None with empty text."""
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[[
          LlmResponse(
              finish_reason=None,
              content=types.Content(
                  role='model',
                  parts=[types.Part.from_text(text='   ')],
              ),
          )
      ]]
  )

  with pytest.raises(AdvisorError, match='returned an empty response'):
    await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
    )


@pytest.mark.asyncio
async def test_call_advisor_response_error_code_raises_and_records_telemetry():
  """Raises AdvisorError on error_code and preserves responses for telemetry."""
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[[
          LlmResponse(
              model_version='gemini-2.5-pro-002',
              error_code='RESOURCE_EXHAUSTED',
              error_message=None,
          )
      ]]
  )

  with mock.patch.object(
      _metrics, 'record_client_operation_duration', autospec=True
  ) as mock_duration:
    with pytest.raises(
        AdvisorError, match='returned error RESOURCE_EXHAUSTED: no message'
    ):
      await call_advisor(
          llm,
          _sample_contents(),
          system_instruction='Advisor system prompt.',
      )
  mock_duration.assert_called_once()
  assert (
      mock_duration.call_args.kwargs['responses'][-1].model_version
      == 'gemini-2.5-pro-002'
  )


@pytest.mark.asyncio
async def test_call_advisor_telemetry_failure_does_not_break_call():
  """Swallows telemetry recording errors so advisor calls still succeed."""
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[[
          LlmResponse(
              finish_reason=types.FinishReason.STOP,
              content=types.Content(
                  role='model',
                  parts=[types.Part.from_text(text='Still works.')],
              ),
          )
      ]]
  )
  with mock.patch.object(
      _metrics,
      'record_client_operation_duration',
      autospec=True,
      side_effect=RuntimeError('OTel exporter error'),
  ):
    result = await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
    )
  assert result.text == 'Still works.'


@pytest.mark.asyncio
async def test_call_advisor_timeout_raises_advisor_error():
  """Raises AdvisorError when the advisor call exceeds timeout_seconds."""
  llm = _FakeAdvisorLlm(delay_seconds=0.2)
  with pytest.raises(AdvisorError, match='timed out after 0.01s'):
    await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
        timeout_seconds=0.01,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_timeout', [0, 0.0, -5.0])
async def test_call_advisor_non_positive_timeout_raises_value_error(
    bad_timeout: float,
):
  """Rejects zero or negative timeout_seconds with ValueError."""
  llm = _FakeAdvisorLlm()
  with pytest.raises(ValueError, match='timeout_seconds must be positive'):
    await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
        timeout_seconds=bad_timeout,
    )


@pytest.mark.asyncio
async def test_call_advisor_transport_timeout_without_timeout_seconds():
  """Formats transport TimeoutError without 'Nones' when timeout is None."""
  llm = _FakeAdvisorLlm(
      scripted_outcomes=[TimeoutError('read timed out on socket')]
  )
  with pytest.raises(
      AdvisorError,
      match=r'Advisor \(fake-advisor-pro\) timed out: read timed out on socket',
  ) as exc_info:
    await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
        timeout_seconds=None,
    )
  assert 'Nones' not in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_advisor_timeout_bounds_total_wall_clock_across_retry():
  """Shares timeout_seconds budget across the initial attempt and retry."""
  llm = _FakeAdvisorLlm(
      delay_seconds=0.04,
      scripted_outcomes=[
          ValueError('thinking_config is unsupported for this model'),
          [
              LlmResponse(
                  finish_reason=types.FinishReason.STOP,
                  content=types.Content(
                      role='model',
                      parts=[types.Part.from_text(text='Too slow.')],
                  ),
              )
          ],
      ],
  )
  with pytest.raises(AdvisorError, match='timed out after 0.06s'):
    await call_advisor(
        llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
        thinking_level=types.ThinkingLevel.HIGH,
        timeout_seconds=0.06,
    )


@pytest.mark.asyncio
async def test_call_advisor_skips_native_telemetry_when_genai_instrumented():
  """Skips native OTel metrics for Gemini when genai OTel lib is active."""
  gemini_llm = _FakeAdvisorLlm(
      model='gemini-2.5-pro',
      scripted_outcomes=[[
          LlmResponse(
              finish_reason=types.FinishReason.STOP,
              content=types.Content(
                  role='model',
                  parts=[types.Part.from_text(text='Gemini advice.')],
              ),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=10,
                  candidates_token_count=5,
                  total_token_count=15,
              ),
          )
      ]],
  )
  non_gemini_llm = _FakeAdvisorLlm(
      model='claude-3-7-sonnet',
      scripted_outcomes=[[
          LlmResponse(
              finish_reason=types.FinishReason.STOP,
              content=types.Content(
                  role='model',
                  parts=[types.Part.from_text(text='Claude advice.')],
              ),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=10,
                  candidates_token_count=5,
                  total_token_count=15,
              ),
          )
      ]],
  )

  with (
      mock.patch.object(
          tracing,
          '_instrumented_with_opentelemetry_instrumentation_google_genai',
          return_value=True,
      ),
      mock.patch.object(
          _metrics, 'record_client_operation_duration', autospec=True
      ) as mock_duration,
      mock.patch.object(
          _metrics, 'record_client_token_usage', autospec=True
      ) as mock_tokens,
  ):
    await call_advisor(
        gemini_llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
    )
    mock_duration.assert_not_called()
    mock_tokens.assert_not_called()

    await call_advisor(
        non_gemini_llm,
        _sample_contents(),
        system_instruction='Advisor system prompt.',
    )
    mock_duration.assert_called_once()
    mock_tokens.assert_called_once()
