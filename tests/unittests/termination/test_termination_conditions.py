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

"""Tests for termination conditions."""

from __future__ import annotations

import asyncio
import time

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.termination.external_termination import ExternalTermination
from google.adk.termination.function_call_termination import FunctionCallTermination
from google.adk.termination.max_iterations_termination import MaxIterationsTermination
from google.adk.termination.termination_condition import AndTerminationCondition
from google.adk.termination.termination_condition import OrTerminationCondition
from google.adk.termination.text_mention_termination import TextMentionTermination
from google.adk.termination.timeout_termination import TimeoutTermination
from google.adk.termination.token_usage_termination import TokenUsageTermination
from google.genai import types
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_event(text: str, author: str = 'agent') -> Event:
  return Event(
      invocation_id='inv-1',
      author=author,
      actions=EventActions(),
      content=types.Content(
          role='model',
          parts=[types.Part(text=text)],
      ),
  )


def _make_token_event(
    total_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> Event:
  return Event(
      invocation_id='inv-1',
      author='agent',
      actions=EventActions(),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=total_tokens,
          prompt_token_count=prompt_tokens,
          candidates_token_count=completion_tokens,
      ),
  )


def _make_function_response_event(function_name: str) -> Event:
  return Event(
      invocation_id='inv-1',
      author='agent',
      actions=EventActions(),
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=function_name,
                      response={'result': 'ok'},
                  )
              )
          ],
      ),
  )


# ---------------------------------------------------------------------------
# MaxIterationsTermination
# ---------------------------------------------------------------------------


class TestMaxIterationsTermination:

  def test_raises_if_not_positive(self):
    with pytest.raises(ValueError):
      MaxIterationsTermination(0)
    with pytest.raises(ValueError):
      MaxIterationsTermination(-1)

  @pytest.mark.asyncio
  async def test_does_not_terminate_before_limit(self):
    condition = MaxIterationsTermination(3)
    result = await condition.check([_make_text_event('hello')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_terminates_at_limit(self):
    condition = MaxIterationsTermination(3)
    await condition.check([_make_text_event('a'), _make_text_event('b')])
    result = await condition.check([_make_text_event('c')])
    assert result is not None
    assert '3' in result.reason
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_does_not_fire_again_after_termination(self):
    condition = MaxIterationsTermination(1)
    await condition.check([_make_text_event('first')])
    assert condition.terminated is True
    second = await condition.check([_make_text_event('second')])
    assert second is None

  @pytest.mark.asyncio
  async def test_reset(self):
    condition = MaxIterationsTermination(1)
    await condition.check([_make_text_event('first')])
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False

    result = await condition.check([_make_text_event('first again')])
    assert result is not None


# ---------------------------------------------------------------------------
# TextMentionTermination
# ---------------------------------------------------------------------------


class TestTextMentionTermination:

  @pytest.mark.asyncio
  async def test_terminates_when_text_found(self):
    condition = TextMentionTermination('TERMINATE')
    result = await condition.check([_make_text_event('Please TERMINATE now.')])
    assert result is not None
    assert 'TERMINATE' in result.reason
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_does_not_terminate_when_absent(self):
    condition = TextMentionTermination('TERMINATE')
    result = await condition.check([_make_text_event('Keep going!')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_respects_sources_filter(self):
    condition = TextMentionTermination('APPROVE', sources=['critic'])

    # Wrong source — should NOT fire
    no_fire = await condition.check([_make_text_event('APPROVE', 'primary')])
    assert no_fire is None

    # Correct source — should fire
    fire = await condition.check([_make_text_event('APPROVE', 'critic')])
    assert fire is not None

  @pytest.mark.asyncio
  async def test_reset(self):
    condition = TextMentionTermination('DONE')
    await condition.check([_make_text_event('DONE')])
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False

    result = await condition.check([_make_text_event('not done yet')])
    assert result is None


# ---------------------------------------------------------------------------
# TokenUsageTermination
# ---------------------------------------------------------------------------


class TestTokenUsageTermination:

  def test_raises_if_no_limit(self):
    with pytest.raises(ValueError):
      TokenUsageTermination()

  @pytest.mark.asyncio
  async def test_terminates_on_total_tokens(self):
    condition = TokenUsageTermination(max_total_tokens=100)
    result = await condition.check([_make_token_event(101, 50, 51)])
    assert result is not None
    assert 'total_tokens' in result.reason
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_terminates_on_prompt_tokens(self):
    condition = TokenUsageTermination(max_prompt_tokens=50)
    result = await condition.check([_make_token_event(60, 55, 5)])
    assert result is not None
    assert 'prompt_tokens' in result.reason

  @pytest.mark.asyncio
  async def test_terminates_on_completion_tokens(self):
    condition = TokenUsageTermination(max_completion_tokens=30)
    result = await condition.check([_make_token_event(40, 5, 35)])
    assert result is not None
    assert 'completion_tokens' in result.reason

  @pytest.mark.asyncio
  async def test_accumulates_across_events(self):
    condition = TokenUsageTermination(max_total_tokens=100)
    await condition.check([_make_token_event(60, 40, 20)])
    assert condition.terminated is False

    result = await condition.check([_make_token_event(50, 30, 20)])
    assert result is not None
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_ignores_events_without_usage(self):
    condition = TokenUsageTermination(max_total_tokens=10)
    result = await condition.check([_make_text_event('no tokens here')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_reset(self):
    condition = TokenUsageTermination(max_total_tokens=100)
    await condition.check([_make_token_event(200, 100, 100)])
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False
    result = await condition.check([_make_token_event(50, 30, 20)])
    assert result is None


# ---------------------------------------------------------------------------
# TimeoutTermination
# ---------------------------------------------------------------------------


class TestTimeoutTermination:

  def test_raises_if_not_positive(self):
    with pytest.raises(ValueError):
      TimeoutTermination(0)
    with pytest.raises(ValueError):
      TimeoutTermination(-5)

  @pytest.mark.asyncio
  async def test_does_not_terminate_before_timeout(self):
    condition = TimeoutTermination(60)
    result = await condition.check([_make_text_event('hello')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_terminates_after_timeout(self):
    condition = TimeoutTermination(0.01)  # 10ms
    # Warm up the start time.
    await condition.check([_make_text_event('trigger start')])
    # Wait slightly longer than the timeout.
    await asyncio.sleep(0.02)

    result = await condition.check([_make_text_event('after timeout')])
    assert result is not None
    assert 'Timeout' in result.reason
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_reset(self):
    condition = TimeoutTermination(0.01)
    await condition.check([_make_text_event('start')])
    await asyncio.sleep(0.02)
    await condition.check([_make_text_event('fires')])
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False
    # After reset a fresh check starts a new timer.
    result = await condition.check([_make_text_event('fresh start')])
    assert result is None


# ---------------------------------------------------------------------------
# FunctionCallTermination
# ---------------------------------------------------------------------------


class TestFunctionCallTermination:

  @pytest.mark.asyncio
  async def test_terminates_on_matching_function(self):
    condition = FunctionCallTermination('approve')
    result = await condition.check([_make_function_response_event('approve')])
    assert result is not None
    assert 'approve' in result.reason
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_does_not_terminate_for_different_function(self):
    condition = FunctionCallTermination('approve')
    result = await condition.check([_make_function_response_event('search')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_does_not_terminate_on_text_only(self):
    condition = FunctionCallTermination('approve')
    result = await condition.check([_make_text_event('approve this')])
    assert result is None

  @pytest.mark.asyncio
  async def test_reset(self):
    condition = FunctionCallTermination('approve')
    await condition.check([_make_function_response_event('approve')])
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False


# ---------------------------------------------------------------------------
# ExternalTermination
# ---------------------------------------------------------------------------


class TestExternalTermination:

  @pytest.mark.asyncio
  async def test_does_not_terminate_before_set(self):
    condition = ExternalTermination()
    result = await condition.check([_make_text_event('anything')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_terminates_after_set(self):
    condition = ExternalTermination()
    condition.set()
    result = await condition.check([_make_text_event('anything')])
    assert result is not None
    assert 'Externally terminated' in result.reason
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_reset(self):
    condition = ExternalTermination()
    condition.set()
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False
    result = await condition.check([_make_text_event('should not fire')])
    assert result is None


# ---------------------------------------------------------------------------
# OrTerminationCondition (.or_())
# ---------------------------------------------------------------------------


class TestOrTerminationCondition:

  @pytest.mark.asyncio
  async def test_terminates_on_first(self):
    condition = MaxIterationsTermination(1).or_(TextMentionTermination('DONE'))
    result = await condition.check([_make_text_event('any')])
    assert result is not None
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_terminates_on_second(self):
    condition = MaxIterationsTermination(100).or_(
        TextMentionTermination('DONE')
    )
    result = await condition.check([_make_text_event('DONE')])
    assert result is not None
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_does_not_terminate_when_neither_fires(self):
    condition = MaxIterationsTermination(100).or_(
        TextMentionTermination('DONE')
    )
    result = await condition.check([_make_text_event('keep going')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_reset_both_children(self):
    condition = MaxIterationsTermination(1).or_(TextMentionTermination('DONE'))
    await condition.check([_make_text_event('fires')])
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False

  def test_is_or_instance(self):
    condition = MaxIterationsTermination(1).or_(TextMentionTermination('X'))
    assert isinstance(condition, OrTerminationCondition)

  @pytest.mark.asyncio
  async def test_pipe_operator(self):
    condition = MaxIterationsTermination(1) | TextMentionTermination('DONE')
    result = await condition.check([_make_text_event('any')])
    assert result is not None
    assert isinstance(condition, OrTerminationCondition)


# ---------------------------------------------------------------------------
# AndTerminationCondition (.and_())
# ---------------------------------------------------------------------------


class TestAndTerminationCondition:

  @pytest.mark.asyncio
  async def test_does_not_terminate_when_only_first_fires(self):
    condition = MaxIterationsTermination(1).and_(TextMentionTermination('DONE'))
    result = await condition.check([_make_text_event('no keyword here')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_does_not_terminate_when_only_second_fires(self):
    condition = MaxIterationsTermination(100).and_(
        TextMentionTermination('DONE')
    )
    result = await condition.check([_make_text_event('DONE')])
    assert result is None
    assert condition.terminated is False

  @pytest.mark.asyncio
  async def test_terminates_when_both_fire(self):
    left = MaxIterationsTermination(1)
    right = TextMentionTermination('DONE')
    condition = left.and_(right)

    result = await condition.check([_make_text_event('DONE')])
    assert result is not None
    assert condition.terminated is True

  @pytest.mark.asyncio
  async def test_reset_both_children(self):
    condition = MaxIterationsTermination(1).and_(TextMentionTermination('DONE'))
    await condition.check([_make_text_event('DONE')])
    assert condition.terminated is True

    await condition.reset()
    assert condition.terminated is False

  def test_is_and_instance(self):
    condition = MaxIterationsTermination(1).and_(TextMentionTermination('X'))
    assert isinstance(condition, AndTerminationCondition)

  @pytest.mark.asyncio
  async def test_ampersand_operator(self):
    condition = MaxIterationsTermination(1) & TextMentionTermination('DONE')
    result = await condition.check([_make_text_event('DONE')])
    assert result is not None
    assert isinstance(condition, AndTerminationCondition)
