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

"""Tests for SemanticArgMatcher."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import re
from typing import Literal

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_args_matcher import _semantic_arg_matcher
from google.adk.tools.tool_args_matcher import ArgMatchResult
from google.adk.tools.tool_args_matcher import BaseArgMatcher
from google.adk.tools.tool_args_matcher import SemanticArgMatcher
from google.adk.tools.tool_args_matcher import ToolArgsMatcher
from google.adk.tools.tool_args_matcher._tool_args_matcher import ExactArgMatcher
from google.genai import types as genai_types
from pydantic import ValidationError
import pytest


async def _match_one(
    matcher: BaseArgMatcher, expected: object, actual: object
) -> ArgMatchResult:
  return await matcher.match(
      tool_name='tool', arg_name='arg', expected=expected, actual=actual
  )


# --- Semantic matcher ---


class _FakeJudge:
  """Answers semantic comparisons with canned text and records each prompt."""

  def __init__(self, *answers: str, error: Exception | None = None):
    self._answers = list(answers)
    self._error = error
    self.requests: list[LlmRequest] = []

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    self.requests.append(llm_request)
    if self._error is not None:
      raise self._error
    yield LlmResponse(
        content=genai_types.Content(
            role='model', parts=[genai_types.Part(text=self._answers.pop(0))]
        )
    )

  @property
  def prompts(self) -> list[str]:
    return [r.contents[0].parts[0].text for r in self.requests]


@pytest.fixture
def judge_factory(monkeypatch: pytest.MonkeyPatch):
  """Makes LLMRegistry hand out the given fake judge, recording model names."""
  requested_models: list[str] = []

  def _install(judge: _FakeJudge) -> list[str]:
    def _new_llm(model: str) -> _FakeJudge:
      requested_models.append(model)
      return judge

    monkeypatch.setattr(
        _semantic_arg_matcher.LLMRegistry, 'new_llm', staticmethod(_new_llm)
    )
    return requested_models

  return _install


async def test_semantic_matches_equal_values_without_model_call(judge_factory):
  """Equal values match without asking the model."""
  judge = _FakeJudge()
  judge_factory(judge)

  result = await _match_one(SemanticArgMatcher(), 'refund', 'refund')

  assert (result.matched, judge.requests) == (True, [])


async def test_semantic_matches_when_model_says_yes(judge_factory):
  """A YES from the model is a match."""
  judge_factory(_FakeJudge('YES'))

  result = await _match_one(
      SemanticArgMatcher(), 'employees in NYC', 'New York City workforce'
  )

  assert (result.matched, result.score) == (True, 1.0)


async def test_semantic_rejects_when_model_says_no(judge_factory):
  """A NO from the model is a mismatch."""
  judge_factory(_FakeJudge('No.'))

  result = await _match_one(SemanticArgMatcher(), 'Paris', 'Lyon')

  assert not result.matched


async def test_semantic_reads_verdict_inside_longer_response(judge_factory):
  """A verdict that is not the first word is still read."""
  judge_factory(_FakeJudge('The answer is YES'))

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert result.matched


async def test_semantic_reads_verdict_after_note_or_not_prefix(judge_factory):
  """Words starting with NO, such as Note or Not, do not count as NO."""
  judge_factory(_FakeJudge('Note: not a mismatch, YES'))

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert result.matched


async def test_semantic_rejects_ambiguous_response(judge_factory):
  """A response with no verdict or score is a mismatch with a reason."""
  judge_factory(_FakeJudge('maybe'))

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert (result.matched, result.score) == (False, 0.0)
  assert 'no verdict or score' in result.reason


async def test_semantic_stores_model_score_and_reason(judge_factory):
  """The score and reason from a JSON judgement are kept on the result."""
  judge_factory(
      _FakeJudge(
          '{"verdict": "YES", "score": 0.82, "reason": "Same city, reworded."}'
      )
  )

  result = await _match_one(SemanticArgMatcher(), 'NYC', 'New York City')

  assert result == ArgMatchResult(
      matched=True, score=0.82, reason='Same city, reworded.'
  )


async def test_semantic_scores_a_mismatch(judge_factory):
  """A NO verdict keeps its partial score."""
  judge_factory(_FakeJudge('{"verdict": "NO", "score": 0.3}'))

  result = await _match_one(SemanticArgMatcher(), 'Paris', 'Paris, Texas')

  assert (result.matched, result.score) == (False, 0.3)


async def test_semantic_reads_json_inside_code_fence(judge_factory):
  """A JSON judgement wrapped in a Markdown code fence is still read."""
  judge_factory(_FakeJudge('```json\n{"verdict": "YES", "score": 0.9}\n```'))

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert (result.matched, result.score) == (True, 0.9)


async def test_semantic_reads_json_after_echoed_dict_in_text(judge_factory):
  """A JSON judgement preceded by echoed braces in prose is still read."""
  judge_factory(
      _FakeJudge(
          'Comparing {"city": "NYC", "note": "no rush"} ...\n'
          '```json\n{"verdict": "YES", "score": 0.9, "reason": "Same."}\n```'
      )
  )

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert result == ArgMatchResult(matched=True, score=0.9, reason='Same.')


async def test_semantic_verdict_without_score_scores_one_or_zero(
    judge_factory,
):
  """A JSON verdict without a score scores 1.0 for YES."""
  judge_factory(_FakeJudge('{"verdict": "YES"}'))

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert (result.matched, result.score) == (True, 1.0)


async def test_semantic_score_without_verdict_decides_by_half(judge_factory):
  """A JSON score without a verdict matches when it is at least 0.5."""
  judge_factory(_FakeJudge('{"score": 0.4}'))

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert (result.matched, result.score) == (False, 0.4)


async def test_semantic_clamps_out_of_range_score(judge_factory):
  """A score outside [0, 1] is clamped instead of failing validation."""
  judge_factory(_FakeJudge('{"verdict": "YES", "score": 7}'))

  result = await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert result.score == 1.0


async def test_semantic_score_feeds_tool_args_mean(judge_factory):
  """The semantic score is averaged into the overall tool call score."""
  judge_factory(_FakeJudge('{"verdict": "YES", "score": 0.6}'))
  matcher = ToolArgsMatcher(per_arg_matchers={'query': SemanticArgMatcher()})

  result = await matcher.match(
      tool_name='search',
      expected={'query': 'NYC', 'limit': 10},
      actual={'query': 'New York', 'limit': 10},
  )

  assert result.matched
  assert result.score == pytest.approx(0.8)


def test_semantic_default_generate_content_config_requests_json():
  """By default the model is asked for JSON output."""
  config = SemanticArgMatcher().generate_content_config

  assert config.response_mime_type == 'application/json'


async def test_semantic_without_model_uses_default_model(judge_factory):
  """Outside Environment Simulation, an unset model means gemini-2.5-flash."""
  judge = _FakeJudge('YES')
  requested_models = judge_factory(judge)

  await _match_one(SemanticArgMatcher(), 'a', 'b')

  assert requested_models == ['gemini-2.5-flash']
  assert judge.requests[0].model == 'gemini-2.5-flash'


class _HangingJudge:
  """Hangs on its first call until cancelled, then answers later calls.

  The first call records its cancellation. Each later call answers with the
  next of `answers`.
  """

  def __init__(self, *answers: str):
    self._answers = list(answers)
    self.calls = 0
    self.cancelled = False

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del llm_request, stream
    self.calls += 1
    if self.calls == 1:
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        self.cancelled = True
        raise
    yield LlmResponse(
        content=genai_types.Content(
            role='model', parts=[genai_types.Part(text=self._answers.pop(0))]
        )
    )


async def test_semantic_times_out_when_model_hangs(judge_factory):
  """A model that does not answer in time raises TimeoutError."""
  judge = _HangingJudge()
  judge_factory(judge)

  with pytest.raises(TimeoutError, match='did not respond within 0.05'):
    await _match_one(SemanticArgMatcher(timeout_seconds=0.05), 'a', 'b')

  assert judge.cancelled


async def test_semantic_timeout_is_not_cached(judge_factory):
  """A timed-out comparison asks the model again next time."""
  judge = _HangingJudge('YES')
  judge_factory(judge)
  matcher = SemanticArgMatcher(timeout_seconds=0.05)
  with pytest.raises(TimeoutError):
    await _match_one(matcher, 'a', 'b')

  result = await _match_one(matcher, 'a', 'b')

  assert (result.matched, judge.calls) == (True, 2)


def test_semantic_default_timeout_is_set():
  """By default, model calls are bounded by a 60 second timeout."""
  assert SemanticArgMatcher().timeout_seconds == 60.0


def test_semantic_timeout_can_be_turned_off():
  """A timeout of None waits for the model indefinitely."""
  assert SemanticArgMatcher(timeout_seconds=None).timeout_seconds is None


def test_semantic_rejects_non_positive_timeout():
  """A zero or negative timeout is rejected."""
  with pytest.raises(ValidationError):
    SemanticArgMatcher(timeout_seconds=0)


async def test_semantic_caches_verdict(judge_factory):
  """The same comparison asks the model only once."""
  judge = _FakeJudge('YES')
  judge_factory(judge)
  matcher = SemanticArgMatcher()

  await _match_one(matcher, 'a', 'b')
  result = await _match_one(matcher, 'a', 'b')

  assert (result.matched, len(judge.requests)) == (True, 1)


async def test_semantic_does_not_cache_ambiguous_response(judge_factory):
  """An unreadable response is asked again rather than remembered."""
  judge = _FakeJudge('maybe', 'YES')
  judge_factory(judge)
  matcher = SemanticArgMatcher()

  await _match_one(matcher, 'a', 'b')
  result = await _match_one(matcher, 'a', 'b')

  assert (result.matched, len(judge.requests)) == (True, 2)


async def test_semantic_model_error_propagates(judge_factory):
  """An error from the model reaches the caller instead of a mismatch."""
  judge_factory(_FakeJudge(error=RuntimeError('quota')))

  with pytest.raises(RuntimeError, match='quota'):
    await _match_one(SemanticArgMatcher(), 'a', 'b')


async def test_semantic_prompt_names_tool_argument_and_values(judge_factory):
  """The prompt tells the model what it is comparing."""
  judge = _FakeJudge('YES')
  judge_factory(judge)

  await SemanticArgMatcher().match(
      tool_name='search', arg_name='query', expected='NYC', actual='New York'
  )

  prompt = judge.prompts[0]
  assert 'Tool: search' in prompt
  assert 'Argument: query' in prompt
  assert 'Expected value: "NYC"' in prompt
  assert 'Actual value: "New York"' in prompt


async def test_semantic_criteria_replace_default_guidance(judge_factory):
  """Custom criteria are sent instead of the default guidance."""
  judge = _FakeJudge('YES')
  judge_factory(judge)

  await _match_one(
      SemanticArgMatcher(criteria='Only the city matters.'), 'a', 'b'
  )

  assert 'Only the city matters.' in judge.prompts[0]
  assert 'synonym' not in judge.prompts[0]


async def test_semantic_uses_configured_model_and_config(judge_factory):
  """The configured model and generate_content_config are used for the call."""
  judge = _FakeJudge('YES')
  requested_models = judge_factory(judge)
  matcher = SemanticArgMatcher(
      model='gemini-2.5-pro',
      generate_content_config=genai_types.GenerateContentConfig(
          temperature=0.5
      ),
  )

  await _match_one(matcher, 'a', 'b')

  assert requested_models == ['gemini-2.5-pro']
  assert judge.requests[0].model == 'gemini-2.5-pro'
  assert judge.requests[0].config.temperature == 0.5


def test_semantic_default_generate_content_config_is_deterministic():
  """By default the model is called at temperature 0."""
  assert SemanticArgMatcher().generate_content_config.temperature == 0.0


def test_semantic_is_not_deterministic():
  """The semantic matcher runs after deterministic matchers."""
  assert not SemanticArgMatcher.is_deterministic


def test_config_loads_semantic_matcher():
  """A config can name the semantic matcher with its options."""
  matcher = ToolArgsMatcher.model_validate({
      'perArgMatchers': {
          'query': {
              'type': 'semantic',
              'model': 'gemini-2.5-pro',
              'criteria': 'Same intent.',
          }
      }
  })

  assert matcher.per_arg_matchers['query'] == SemanticArgMatcher(
      model='gemini-2.5-pro', criteria='Same intent.'
  )


async def test_short_circuit_skips_semantic_after_deterministic_miss(
    judge_factory,
):
  """A deterministic mismatch saves the semantic model call."""
  judge = _FakeJudge()
  judge_factory(judge)
  matcher = ToolArgsMatcher(per_arg_matchers={'query': SemanticArgMatcher()})

  result = await matcher.match(
      tool_name='search',
      expected={'query': 'NYC', 'limit': 10},
      actual={'query': 'New York', 'limit': 5},
      short_circuit=True,
  )

  assert (result.matched, judge.requests) == (False, [])


# --- Semantic matcher: parallelism inside ToolArgsMatcher ---


class _ConcurrencyJudge:
  """Answers per argument after a delay, recording how calls overlap.

  The argument is read from the prompt, so each concurrent call can be given
  its own answer and delay, and completion order can differ from start order.
  """

  def __init__(
      self,
      answers: dict[str, str],
      delays: dict[str, float] | None = None,
      default_delay: float = 0.05,
  ):
    self._answers = answers
    self._delays = delays or {}
    self._default_delay = default_delay
    self.in_flight = 0
    self.max_in_flight = 0
    self.started: list[str] = []
    self.finished: list[str] = []

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    prompt = llm_request.contents[0].parts[0].text
    arg_name = re.search(r'Argument: (\S+)', prompt).group(1)
    self.started.append(arg_name)
    self.in_flight += 1
    self.max_in_flight = max(self.max_in_flight, self.in_flight)
    try:
      await asyncio.sleep(self._delays.get(arg_name, self._default_delay))
    finally:
      self.in_flight -= 1
    self.finished.append(arg_name)
    yield LlmResponse(
        content=genai_types.Content(
            role='model',
            parts=[genai_types.Part(text=self._answers[arg_name])],
        )
    )


async def test_semantic_args_are_judged_concurrently(judge_factory):
  """Every semantic argument of one call is sent to the model at once."""
  judge = _ConcurrencyJudge(answers={'a': 'YES', 'b': 'YES', 'c': 'YES'})
  judge_factory(judge)
  matcher = ToolArgsMatcher(default_matcher=SemanticArgMatcher())

  result = await matcher.match(
      tool_name='tool',
      expected={'a': 'x1', 'b': 'y1', 'c': 'z1'},
      actual={'a': 'x2', 'b': 'y2', 'c': 'z2'},
  )

  assert result.matched
  assert judge.max_in_flight == 3


async def test_concurrent_results_map_back_to_their_arguments(judge_factory):
  """Results land on the right argument when calls finish out of order."""
  # The first argument is the slowest, so completion order is reversed.
  judge = _ConcurrencyJudge(
      answers={
          'a': '{"verdict": "NO", "score": 0.1}',
          'b': '{"verdict": "YES", "score": 0.9}',
          'c': '{"verdict": "YES", "score": 0.7}',
      },
      delays={'a': 0.15, 'b': 0.1, 'c': 0.05},
  )
  judge_factory(judge)
  matcher = ToolArgsMatcher(default_matcher=SemanticArgMatcher())

  result = await matcher.match(
      tool_name='tool',
      expected={'a': 'x1', 'b': 'y1', 'c': 'z1'},
      actual={'a': 'x2', 'b': 'y2', 'c': 'z2'},
  )

  assert judge.finished == ['c', 'b', 'a']
  assert list(result.per_arg) == ['a', 'b', 'c']
  assert [r.matched for r in result.per_arg.values()] == [False, True, True]
  assert [r.score for r in result.per_arg.values()] == [0.1, 0.9, 0.7]
  assert result.score == pytest.approx((0.1 + 0.9 + 0.7) / 3)


async def test_deterministic_args_finish_before_any_model_call(judge_factory):
  """Cheap comparisons all run before the first semantic call starts."""
  log: list[str] = []

  class _LoggingExactMatcher(BaseArgMatcher):
    type: Literal['logging_exact'] = 'logging_exact'

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      log.append(f'deterministic:{arg_name}')
      return ArgMatchResult(matched=expected == actual)

  class _LoggingJudge(_ConcurrencyJudge):

    async def generate_content_async(self, llm_request, stream=False):
      prompt = llm_request.contents[0].parts[0].text
      log.append('model:' + re.search(r'Argument: (\S+)', prompt).group(1))
      async for response in super().generate_content_async(llm_request):
        yield response

  judge_factory(_LoggingJudge(answers={'query': 'YES', 'topic': 'YES'}))
  matcher = ToolArgsMatcher(
      default_matcher=_LoggingExactMatcher(),
      per_arg_matchers={
          'query': SemanticArgMatcher(),
          'topic': SemanticArgMatcher(),
      },
  )

  await matcher.match(
      tool_name='tool',
      # Semantic arguments are listed first to show the order is not
      # simply the order of `expected`.
      expected={'query': 'q1', 'topic': 't1', 'id': 1, 'limit': 5},
      actual={'query': 'q2', 'topic': 't2', 'id': 1, 'limit': 5},
  )

  assert log[:2] == ['deterministic:id', 'deterministic:limit']
  assert sorted(log[2:]) == ['model:query', 'model:topic']


async def test_short_circuit_makes_no_model_calls_after_cheap_miss(
    judge_factory,
):
  """With short_circuit, a deterministic miss cancels all semantic work."""
  judge = _ConcurrencyJudge(answers={'a': 'YES', 'b': 'YES'})
  judge_factory(judge)
  matcher = ToolArgsMatcher(
      default_matcher=SemanticArgMatcher(),
      per_arg_matchers={'id': ExactArgMatcher()},
  )

  result = await matcher.match(
      tool_name='tool',
      expected={'a': 'x1', 'b': 'y1', 'id': 1},
      actual={'a': 'x2', 'b': 'y2', 'id': 2},
      short_circuit=True,
  )

  assert not result.matched
  assert judge.started == []
