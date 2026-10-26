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

"""Tests for ToolArgsMatcher and the built-in argument matchers."""

from __future__ import annotations

import asyncio
from typing import Literal

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools.tool_args_matcher import _tool_args_matcher
from google.adk.tools.tool_args_matcher import ArgMatchResult
from google.adk.tools.tool_args_matcher import BaseArgMatcher
from google.adk.tools.tool_args_matcher import register_arg_matcher
from google.adk.tools.tool_args_matcher import ToolArgsMatcher
from google.adk.tools.tool_args_matcher._tool_args_matcher import CasefoldArgMatcher
from google.adk.tools.tool_args_matcher._tool_args_matcher import ContainsArgMatcher
from google.adk.tools.tool_args_matcher._tool_args_matcher import ExactArgMatcher
from google.adk.tools.tool_args_matcher._tool_args_matcher import IgnoreArgMatcher
from google.adk.tools.tool_args_matcher._tool_args_matcher import NumericArgMatcher
from pydantic import PrivateAttr
from pydantic import ValidationError
import pytest

# --- Fixtures ---


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps matchers registered by one test out of the others."""
  monkeypatch.setattr(
      _tool_args_matcher,
      '_ARG_MATCHER_TYPES',
      dict(_tool_args_matcher._ARG_MATCHER_TYPES),
  )


class _RecordingMatcher(BaseArgMatcher):
  """A non-deterministic matcher that records the arguments it compares."""

  type: Literal['recording'] = 'recording'
  is_deterministic = False

  result: bool = True

  _calls: list[str] = PrivateAttr(default_factory=list)

  @property
  def calls(self) -> list[str]:
    return self._calls

  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    self._calls.append(arg_name)
    return ArgMatchResult(matched=self.result)


async def _match_one(
    matcher: BaseArgMatcher, expected: object, actual: object
) -> ArgMatchResult:
  return await matcher.match(
      tool_name='tool', arg_name='arg', expected=expected, actual=actual
  )


# --- ArgMatchResult ---


def test_result_score_defaults_to_one_when_matched():
  """A matched result without an explicit score scores 1.0."""
  assert ArgMatchResult(matched=True).score == 1.0


def test_result_score_defaults_to_zero_when_not_matched():
  """An unmatched result without an explicit score scores 0.0."""
  assert ArgMatchResult(matched=False).score == 0.0


def test_result_keeps_explicit_score():
  """An explicit score is kept instead of the default from `matched`."""
  assert ArgMatchResult(matched=True, score=0.7).score == 0.7


def test_result_score_follows_parsed_matched_value():
  """The default score follows `matched` after parsing, not the raw input."""
  result = ArgMatchResult.model_validate({'matched': 'false'})

  assert (result.matched, result.score) == (False, 0.0)


def test_result_rejects_score_above_one():
  """A score outside [0, 1] is rejected."""
  with pytest.raises(ValidationError):
    ArgMatchResult(matched=True, score=1.5)


# --- Built-in matchers ---


async def test_exact_matches_equal_values():
  """The exact matcher matches equal values."""
  result = await _match_one(ExactArgMatcher(), {'a': [1]}, {'a': [1]})

  assert result.matched


async def test_exact_rejects_different_values():
  """The exact matcher rejects values that differ only in case."""
  result = await _match_one(ExactArgMatcher(), 'Paris', 'paris')

  assert not result.matched


async def test_casefold_ignores_case_and_surrounding_whitespace():
  """The casefold matcher matches strings that differ in case and padding."""
  result = await _match_one(CasefoldArgMatcher(), 'Straße', '  STRASSE ')

  assert result.matched


async def test_casefold_rejects_different_strings():
  """The casefold matcher rejects strings with different letters."""
  result = await _match_one(CasefoldArgMatcher(), 'Paris', 'Lyon')

  assert not result.matched


async def test_casefold_compares_non_strings_for_equality():
  """The casefold matcher does not coerce a number to a string."""
  result = await _match_one(CasefoldArgMatcher(), 1, '1')

  assert not result.matched


async def test_numeric_matches_within_tolerance():
  """The numeric matcher matches numbers within the tolerance."""
  result = await _match_one(NumericArgMatcher(tolerance=0.1), 1.0, 1.05)

  assert result.matched


async def test_numeric_rejects_outside_tolerance():
  """The numeric matcher rejects numbers further apart than the tolerance."""
  result = await _match_one(NumericArgMatcher(tolerance=0.1), 1.0, 1.2)

  assert not result.matched


async def test_numeric_default_tolerance_requires_equal_numbers():
  """Without a tolerance, the numeric matcher still treats 1 and 1.0 alike."""
  result = await _match_one(NumericArgMatcher(), 1, 1.0)

  assert result.matched


async def test_numeric_rejects_booleans():
  """The numeric matcher does not treat booleans as numbers."""
  result = await _match_one(NumericArgMatcher(), 1, True)

  assert not result.matched


async def test_numeric_explains_non_numeric_values():
  """The numeric matcher names the types when a value is not a number."""
  result = await _match_one(NumericArgMatcher(), 1, '1')

  assert result.reason == 'Both values must be numbers, got int and str.'


def test_numeric_rejects_negative_tolerance():
  """A negative tolerance is rejected."""
  with pytest.raises(ValidationError):
    NumericArgMatcher(tolerance=-1)


async def test_contains_matches_substring():
  """The contains matcher matches when the expected text is in the actual."""
  result = await _match_one(
      ContainsArgMatcher(), 'refund', 'please refund order 123'
  )

  assert result.matched


async def test_contains_rejects_missing_substring():
  """The contains matcher rejects actual text without the expected text."""
  result = await _match_one(ContainsArgMatcher(), 'refund', 'cancel order')

  assert not result.matched


async def test_contains_compares_non_strings_for_equality():
  """The contains matcher does not search inside non-string values."""
  result = await _match_one(ContainsArgMatcher(), 1, [1, 2])

  assert not result.matched


async def test_ignore_matches_any_value():
  """The ignore matcher matches regardless of the actual value."""
  result = await _match_one(IgnoreArgMatcher(), 'abc', None)

  assert result.matched


# --- ToolArgsMatcher: matching ---


async def test_default_matcher_compares_unlisted_args_exactly():
  """Without config, arguments are compared for exact equality."""
  result = await ToolArgsMatcher().match(
      tool_name='tool', expected={'city': 'Paris'}, actual={'city': 'paris'}
  )

  assert not result.matched


async def test_per_arg_matcher_overrides_default():
  """A matcher listed for an argument is used instead of the default."""
  matcher = ToolArgsMatcher(per_arg_matchers={'city': CasefoldArgMatcher()})

  result = await matcher.match(
      tool_name='tool', expected={'city': 'Paris'}, actual={'city': 'paris'}
  )

  assert result.matched


async def test_missing_arg_does_not_match():
  """An expected argument absent from the call fails with score 0."""
  matcher = ToolArgsMatcher(default_matcher=IgnoreArgMatcher())

  result = await matcher.match(
      tool_name='tool', expected={'city': 'Paris'}, actual={}
  )

  assert not result.matched
  assert result.per_arg['city'].score == 0.0


async def test_empty_expected_matches_any_call():
  """With no expected arguments, any call matches with score 1."""
  result = await ToolArgsMatcher().match(
      tool_name='tool', expected={}, actual={'city': 'Paris'}
  )

  assert (result.matched, result.score) == (True, 1.0)


async def test_empty_expected_rejects_extra_args_when_disallowed():
  """With no expected arguments, extra args fail if they are not allowed."""
  matcher = ToolArgsMatcher(allow_extra_args=False)

  result = await matcher.match(
      tool_name='tool', expected={}, actual={'city': 'Paris'}
  )

  assert not result.matched


async def test_empty_expected_scores_one_despite_disallowed_extra_args():
  """Disallowed extra arguments do not lower the score of an empty match."""
  matcher = ToolArgsMatcher(allow_extra_args=False)

  result = await matcher.match(
      tool_name='tool', expected={}, actual={'city': 'Paris'}
  )

  assert result.score == 1.0


async def test_call_fails_when_one_arg_fails():
  """Every expected argument must match for the call to match."""
  result = await ToolArgsMatcher().match(
      tool_name='tool',
      expected={'city': 'Paris', 'days': 3},
      actual={'city': 'Paris', 'days': 4},
  )

  assert not result.matched


async def test_extra_args_are_allowed_by_default():
  """Arguments the expectation does not name are ignored by default."""
  result = await ToolArgsMatcher().match(
      tool_name='tool',
      expected={'city': 'Paris'},
      actual={'city': 'Paris', 'days': 4},
  )

  assert result.matched


async def test_disallowed_extra_args_fail_match_but_keep_score():
  """Disallowed extra arguments fail the match without lowering the score."""
  matcher = ToolArgsMatcher(allow_extra_args=False)

  result = await matcher.match(
      tool_name='tool',
      expected={'city': 'Paris'},
      actual={'city': 'Paris', 'days': 4},
  )

  assert (result.matched, result.score) == (False, 1.0)


async def test_score_is_mean_of_arg_scores():
  """The overall score averages the per-argument scores."""
  result = await ToolArgsMatcher().match(
      tool_name='tool',
      expected={'a': 1, 'b': 2, 'c': 3, 'd': 4},
      actual={'a': 1, 'b': 2, 'c': 3, 'd': 0},
  )

  assert result.score == 0.75


async def test_per_arg_results_follow_expected_order():
  """Per-argument results are listed in the order of `expected`."""
  matcher = ToolArgsMatcher(per_arg_matchers={'a': _RecordingMatcher()})

  result = await matcher.match(
      tool_name='tool',
      expected={'a': 1, 'b': 2, 'c': 3},
      actual={'c': 3, 'b': 2, 'a': 1},
  )

  assert list(result.per_arg) == ['a', 'b', 'c']


async def test_matcher_receives_tool_and_arg_names():
  """Each matcher is told which tool and argument it is comparing."""
  seen = []

  class _NameRecordingMatcher(BaseArgMatcher):
    type: Literal['names'] = 'names'

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      seen.append((tool_name, arg_name, expected, actual))
      return ArgMatchResult(matched=True)

  matcher = ToolArgsMatcher(default_matcher=_NameRecordingMatcher())

  await matcher.match(
      tool_name='book_flight', expected={'city': 'Paris'}, actual={'city': 'P'}
  )

  assert seen == [('book_flight', 'city', 'Paris', 'P')]


async def test_matcher_exception_propagates():
  """An exception raised by a matcher reaches the caller."""

  class _FailingMatcher(BaseArgMatcher):
    type: Literal['failing'] = 'failing'

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      raise RuntimeError('backend down')

  matcher = ToolArgsMatcher(default_matcher=_FailingMatcher())

  with pytest.raises(RuntimeError, match='backend down'):
    await matcher.match(tool_name='tool', expected={'a': 1}, actual={'a': 1})


# --- ToolArgsMatcher: non-deterministic matchers ---


async def test_short_circuit_skips_non_deterministic_after_failure():
  """Once a deterministic argument fails, short-circuit skips the rest."""
  recording = _RecordingMatcher()
  matcher = ToolArgsMatcher(per_arg_matchers={'query': recording})

  result = await matcher.match(
      tool_name='tool',
      expected={'query': 'refund', 'id': 1},
      actual={'query': 'refund please', 'id': 2},
      short_circuit=True,
  )

  assert (result.matched, recording.calls) == (False, [])


async def test_short_circuit_skips_non_deterministic_on_extra_args():
  """Disallowed extra arguments decide the result before any comparison."""
  recording = _RecordingMatcher()
  matcher = ToolArgsMatcher(allow_extra_args=False, default_matcher=recording)

  result = await matcher.match(
      tool_name='tool',
      expected={'query': 'refund'},
      actual={'query': 'refund', 'id': 1},
      short_circuit=True,
  )

  assert (result.matched, recording.calls) == (False, [])


async def test_short_circuit_score_covers_only_compared_args():
  """When short-circuit skips every comparison, the score is 0.0."""
  matcher = ToolArgsMatcher(
      allow_extra_args=False, default_matcher=_RecordingMatcher()
  )

  result = await matcher.match(
      tool_name='tool',
      expected={'query': 'refund'},
      actual={'query': 'refund', 'id': 1},
      short_circuit=True,
  )

  assert (result.score, result.per_arg) == (0.0, {})


async def test_short_circuit_runs_non_deterministic_when_undecided():
  """Short-circuit still runs the remaining matchers when they decide."""
  recording = _RecordingMatcher(result=False)
  matcher = ToolArgsMatcher(per_arg_matchers={'query': recording})

  result = await matcher.match(
      tool_name='tool',
      expected={'query': 'refund', 'id': 1},
      actual={'query': 'refund please', 'id': 1},
      short_circuit=True,
  )

  assert (result.matched, recording.calls) == (False, ['query'])


async def test_without_short_circuit_every_arg_is_compared():
  """Without short-circuit, every argument is compared after a failure."""
  recording = _RecordingMatcher()
  matcher = ToolArgsMatcher(per_arg_matchers={'query': recording})

  result = await matcher.match(
      tool_name='tool',
      expected={'query': 'refund', 'id': 1},
      actual={'query': 'refund please', 'id': 2},
  )

  assert list(result.per_arg) == ['query', 'id']
  assert result.score == 0.5


async def test_non_deterministic_matchers_run_concurrently():
  """Non-deterministic matchers are awaited together, not one by one."""
  barrier = asyncio.Barrier(2)

  class _BarrierMatcher(BaseArgMatcher):
    type: Literal['barrier'] = 'barrier'
    is_deterministic = False

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      await barrier.wait()
      return ArgMatchResult(matched=True)

  matcher = ToolArgsMatcher(default_matcher=_BarrierMatcher())

  result = await asyncio.wait_for(
      matcher.match(
          tool_name='tool', expected={'a': 1, 'b': 2}, actual={'a': 1, 'b': 2}
      ),
      timeout=5,
  )

  assert result.matched


async def test_failing_matcher_cancels_other_comparisons():
  """When one comparison raises, the comparisons still running are cancelled.

  Setup: argument 'a' fails as soon as 'b' has started; 'b' waits forever.
  Act: compare both arguments.
  Assert: the error reaches the caller and 'b' was cancelled, not left running.
  """
  b_started = asyncio.Event()
  b_cancelled = asyncio.Event()

  class _FailFastMatcher(BaseArgMatcher):
    type: Literal['fail_fast'] = 'fail_fast'
    is_deterministic = False

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      if arg_name == 'a':
        await b_started.wait()
        raise RuntimeError('judge down')
      b_started.set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        b_cancelled.set()
        raise
      return ArgMatchResult(matched=True)

  matcher = ToolArgsMatcher(default_matcher=_FailFastMatcher())

  with pytest.raises(RuntimeError, match='judge down'):
    await asyncio.wait_for(
        matcher.match(
            tool_name='tool', expected={'a': 1, 'b': 2}, actual={'a': 1, 'b': 2}
        ),
        timeout=5,
    )

  assert b_cancelled.is_set()


# --- ToolArgsMatcher: config ---


def test_config_loads_registered_matchers_by_type():
  """A camelCase config builds the matchers named by `type`."""
  matcher = ToolArgsMatcher.model_validate({
      'defaultMatcher': {'type': 'casefold'},
      'perArgMatchers': {'amount': {'type': 'numeric', 'tolerance': 0.5}},
      'allowExtraArgs': False,
  })

  assert matcher == ToolArgsMatcher(
      default_matcher=CasefoldArgMatcher(),
      per_arg_matchers={'amount': NumericArgMatcher(tolerance=0.5)},
      allow_extra_args=False,
  )


def test_config_accepts_snake_case_keys():
  """A snake_case config loads the same as a camelCase one."""
  matcher = ToolArgsMatcher.model_validate(
      {'per_arg_matchers': {'city': {'type': 'casefold'}}}
  )

  assert matcher.per_arg_matchers == {'city': CasefoldArgMatcher()}


def test_config_round_trips_through_dump():
  """Dumping by alias and loading again gives an equal matcher."""
  matcher = ToolArgsMatcher(
      per_arg_matchers={'amount': NumericArgMatcher(tolerance=0.5)},
      allow_extra_args=False,
  )

  restored = ToolArgsMatcher.model_validate(matcher.model_dump(by_alias=True))

  assert restored == matcher


def test_config_dump_includes_subclass_fields():
  """Dumping keeps fields that only a matcher subclass defines."""
  matcher = ToolArgsMatcher(default_matcher=NumericArgMatcher(tolerance=0.5))

  dumped = matcher.model_dump(by_alias=True)

  assert dumped['defaultMatcher'] == {'type': 'numeric', 'tolerance': 0.5}


def test_config_rejects_unknown_matcher_type():
  """A config naming an unregistered type fails and lists known types."""
  with pytest.raises(ValidationError, match="Unknown arg matcher type 'fuzzy'"):
    ToolArgsMatcher.model_validate({'defaultMatcher': {'type': 'fuzzy'}})


def test_config_rejects_non_string_matcher_type():
  """A config whose `type` is not a string fails validation."""
  with pytest.raises(ValidationError, match='Unknown arg matcher type'):
    ToolArgsMatcher.model_validate({'defaultMatcher': {'type': ['exact']}})


def test_config_rejects_unknown_matcher_field():
  """A misspelled matcher field is rejected instead of ignored."""
  with pytest.raises(ValidationError):
    ToolArgsMatcher.model_validate(
        {'defaultMatcher': {'type': 'numeric', 'tolerence': 0.5}}
    )


def test_config_rejects_unknown_top_level_field():
  """A misspelled ToolArgsMatcher field is rejected instead of ignored."""
  with pytest.raises(ValidationError):
    ToolArgsMatcher.model_validate({'requireAll': True})


async def test_unregistered_matcher_instance_is_accepted():
  """A matcher instance can be used without registering its class."""
  recording = _RecordingMatcher()
  matcher = ToolArgsMatcher(default_matcher=recording)

  await matcher.match(tool_name='tool', expected={'a': 1}, actual={'a': 1})

  assert recording.calls == ['a']


# --- Registry ---


def test_registered_custom_matcher_loads_from_config():
  """A registered custom matcher can be named in config."""

  @register_arg_matcher
  class _PrefixMatcher(BaseArgMatcher):
    type: Literal['prefix'] = 'prefix'

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      return ArgMatchResult(matched=str(actual).startswith(str(expected)))

  matcher = ToolArgsMatcher.model_validate(
      {'defaultMatcher': {'type': 'prefix'}}
  )

  assert isinstance(matcher.default_matcher, _PrefixMatcher)


def test_register_rejects_non_matcher_class():
  """Only BaseArgMatcher subclasses can be registered."""
  with pytest.raises(TypeError):
    register_arg_matcher(dict)


def test_register_rejects_matcher_without_type_default():
  """A matcher class must name itself with a default `type`."""

  class _UnnamedMatcher(BaseArgMatcher):

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      return ArgMatchResult(matched=True)

  with pytest.raises(ValueError, match='non-empty string default'):
    register_arg_matcher(_UnnamedMatcher)


def test_register_rejects_taken_type_name():
  """A second class cannot take a type name that is already registered."""

  class _UpperMatcher(BaseArgMatcher):
    type: Literal['upper'] = 'upper'

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      return ArgMatchResult(matched=True)

  class _OtherUpperMatcher(_UpperMatcher):
    pass

  register_arg_matcher(_UpperMatcher)

  with pytest.raises(ValueError, match='already registered by _UpperMatcher'):
    register_arg_matcher(_OtherUpperMatcher)


def test_register_rejects_reserved_type_name():
  """A matcher outside ADK cannot take a name reserved for a built-in."""

  class _CustomSemanticMatcher(BaseArgMatcher):
    type: Literal['semantic'] = 'semantic'

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      return ArgMatchResult(matched=True)

  with pytest.raises(ValueError, match="'semantic' is reserved"):
    register_arg_matcher(_CustomSemanticMatcher)


def test_register_same_class_twice_is_allowed():
  """Registering a class again returns it without error."""
  assert register_arg_matcher(ExactArgMatcher) is ExactArgMatcher


# --- Feature gating ---


def test_matcher_cannot_be_created_when_feature_is_disabled() -> None:
  """Creating a matcher raises while the TOOL_ARGS_MATCHER feature is off."""
  with temporary_feature_override(FeatureName.TOOL_ARGS_MATCHER, False):
    with pytest.raises(RuntimeError, match='TOOL_ARGS_MATCHER is not enabled'):
      ToolArgsMatcher()
