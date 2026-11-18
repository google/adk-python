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

"""Pluggable comparison of expected and actual tool call arguments.

A `ToolArgsMatcher` compares the arguments of a tool call against an expected
set of arguments, one argument at a time, using a `BaseArgMatcher` per
argument. Matchers are Pydantic models discriminated by their `type` field, so
a `ToolArgsMatcher` can be written in a JSON or YAML config and loaded with
`ToolArgsMatcher.model_validate`.

Custom matchers, for example one that asks an LLM whether two values mean the
same thing, subclass `BaseArgMatcher` and are made loadable from config with
`register_arg_matcher`.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Mapping
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import TypeVar

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import SerializeAsAny
from typing_extensions import override
from typing_extensions import TypeGuard

from ...features import experimental
from ...features import FeatureName
from ...utils._serialized_base_model import SerializedBaseModel

_MATCHER_MODEL_CONFIG = ConfigDict(
    extra='forbid',
    alias_generator=alias_generators.to_camel,
    populate_by_name=True,
)


class ArgMatchResult(SerializedBaseModel):
  """The result of comparing one expected argument value to an actual one."""

  matched: bool
  """Whether the actual value satisfies the expected value."""

  score: float = Field(default=0.0, ge=0.0, le=1.0)
  """How well the actual value matches, from 0.0 to 1.0.

  Defaults to 1.0 when `matched` is true and 0.0 otherwise.
  """

  reason: str | None = None
  """A human-readable explanation of the result, if the matcher has one."""

  @model_validator(mode='after')
  def _default_score_from_matched(self) -> ArgMatchResult:
    """Fills in `score` from `matched` when the caller leaves it out.

    This runs after validation, so it reads `matched` as the parsed boolean
    rather than the raw input, in which a value such as the string 'false'
    would count as true.
    """
    if 'score' not in self.model_fields_set:
      self.score = 1.0 if self.matched else 0.0
    return self


class ToolArgsMatchResult(SerializedBaseModel):
  """The result of comparing all expected arguments of a tool call."""

  matched: bool
  """Whether the tool call arguments satisfy the expectation as a whole."""

  score: float = Field(ge=0.0, le=1.0)
  """The mean score of the arguments that were compared.

  1.0 when nothing was expected. Extra arguments in the call never lower it,
  except when `short_circuit` skips comparisons: the score then covers only
  the arguments that were compared, and is 0.0 when none were.
  """

  per_arg: dict[str, ArgMatchResult] = Field(default_factory=dict)
  """The result for each compared argument, in the order of `expected`.

  An argument whose comparison was skipped by `short_circuit` is absent.
  """


@experimental(FeatureName.TOOL_ARGS_MATCHER)
class BaseArgMatcher(BaseModel, abc.ABC):
  """Compares one expected argument value with the actual value.

  A subclass narrows `type` to a `Literal` with a default, which names the
  matcher in config, and implements `match`. Call `register_arg_matcher` on the
  subclass to make it loadable from config.

  The names `exact`, `casefold`, `numeric`, `contains`, `ignore` and `semantic`
  are reserved for matchers that ship with ADK, and `register_arg_matcher`
  rejects them for other classes.
  """

  model_config = _MATCHER_MODEL_CONFIG

  type: str
  """The name that identifies this matcher in config."""

  is_deterministic: ClassVar[bool] = True
  """Whether `match` returns the same result for the same input.

  `ToolArgsMatcher` compares arguments with deterministic matchers first and
  runs the others concurrently afterwards, so that a cheap mismatch can skip an
  expensive comparison. A matcher that calls a model or a remote service sets
  this to False.
  """

  @abc.abstractmethod
  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    """Compares an expected argument value with the actual value.

    Args:
      tool_name: The name of the tool that was called.
      arg_name: The name of the argument being compared.
      expected: The expected value of the argument.
      actual: The value the tool was called with.

    Returns:
      The result of the comparison.
    """


_ArgMatcherT = TypeVar('_ArgMatcherT', bound=BaseArgMatcher)

_ARG_MATCHER_TYPES: dict[str, type[BaseArgMatcher]] = {}

_RESERVED_ARG_MATCHER_TYPES = frozenset(
    {'exact', 'casefold', 'numeric', 'contains', 'ignore', 'semantic'}
)
"""Matcher names that only matchers shipped with ADK may register.

A name is reserved before its matcher ships, so that a custom matcher
registered under it today does not collide with the built-in one later.
"""

_ADK_MATCHER_MODULE_PREFIX = __name__.rpartition('.')[0] + '.'
"""The module prefix of the package that the ADK matchers are defined in."""


def register_arg_matcher(
    matcher_cls: type[_ArgMatcherT],
) -> type[_ArgMatcherT]:
  """Makes a `BaseArgMatcher` subclass loadable from config by its `type`.

  Registering the same class twice has no effect. This function can be used as
  a class decorator.

  Args:
    matcher_cls: The matcher class to register.

  Returns:
    The matcher class, unchanged.

  Raises:
    TypeError: If `matcher_cls` is not a subclass of `BaseArgMatcher`.
    ValueError: If `matcher_cls` has no default string for `type`, its `type`
      is reserved for a matcher that ships with ADK, or another class is
      already registered under the same `type`.
  """
  if not (
      isinstance(matcher_cls, type) and issubclass(matcher_cls, BaseArgMatcher)
  ):
    raise TypeError(
        f'Expected a subclass of BaseArgMatcher, got {matcher_cls!r}.'
    )
  type_field = matcher_cls.model_fields.get('type')
  matcher_type = type_field.default if type_field else None
  if not isinstance(matcher_type, str) or not matcher_type:
    raise ValueError(
        f'{matcher_cls.__name__} must give the `type` field a non-empty'
        ' string default to be registered.'
    )
  if matcher_type in _RESERVED_ARG_MATCHER_TYPES and not (
      matcher_cls.__module__.startswith(_ADK_MATCHER_MODULE_PREFIX)
  ):
    raise ValueError(
        f'Arg matcher type {matcher_type!r} is reserved for matchers that'
        f' ship with ADK. Give {matcher_cls.__name__} another `type`.'
    )
  registered_cls = _ARG_MATCHER_TYPES.get(matcher_type)
  if registered_cls is not None and registered_cls is not matcher_cls:
    raise ValueError(
        f'Arg matcher type {matcher_type!r} is already registered by'
        f' {registered_cls.__name__}.'
    )
  _ARG_MATCHER_TYPES[matcher_type] = matcher_cls
  return matcher_cls


def _to_arg_matcher(value: Any) -> Any:
  """Builds the registered matcher that a config mapping names by `type`."""
  if not isinstance(value, Mapping):
    return value
  matcher_type = value.get('type')
  matcher_cls = (
      _ARG_MATCHER_TYPES.get(matcher_type)
      if isinstance(matcher_type, str)
      else None
  )
  if matcher_cls is None:
    raise ValueError(
        f'Unknown arg matcher type {matcher_type!r}. Registered types:'
        f' {sorted(_ARG_MATCHER_TYPES)}.'
    )
  return matcher_cls.model_validate(value)


@register_arg_matcher
class ExactArgMatcher(BaseArgMatcher):
  """Matches when the actual value equals the expected value."""

  type: Literal['exact'] = 'exact'

  @override
  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    return ArgMatchResult(matched=expected == actual)


@register_arg_matcher
class CasefoldArgMatcher(BaseArgMatcher):
  """Matches strings that are equal ignoring case and surrounding whitespace.

  Values that are not both strings are compared for equality.
  """

  type: Literal['casefold'] = 'casefold'

  @override
  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    if isinstance(expected, str) and isinstance(actual, str):
      return ArgMatchResult(
          matched=expected.strip().casefold() == actual.strip().casefold()
      )
    return ArgMatchResult(matched=expected == actual)


def _is_number(value: object) -> TypeGuard[int | float]:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


@register_arg_matcher
class NumericArgMatcher(BaseArgMatcher):
  """Matches numbers that differ by at most `tolerance`.

  Booleans and other non-numeric values never match.
  """

  type: Literal['numeric'] = 'numeric'

  tolerance: float = Field(default=0.0, ge=0.0)
  """The largest absolute difference that still counts as a match."""

  @override
  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    if _is_number(expected) and _is_number(actual):
      return ArgMatchResult(matched=abs(expected - actual) <= self.tolerance)
    return ArgMatchResult(
        matched=False,
        reason=(
            'Both values must be numbers, got'
            f' {type(expected).__name__} and {type(actual).__name__}.'
        ),
    )


@register_arg_matcher
class ContainsArgMatcher(BaseArgMatcher):
  """Matches when the expected string occurs within the actual string.

  Values that are not both strings are compared for equality.
  """

  type: Literal['contains'] = 'contains'

  @override
  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    if isinstance(expected, str) and isinstance(actual, str):
      return ArgMatchResult(matched=expected in actual)
    return ArgMatchResult(matched=expected == actual)


@register_arg_matcher
class IgnoreArgMatcher(BaseArgMatcher):
  """Matches any actual value, as long as the argument is present."""

  type: Literal['ignore'] = 'ignore'

  @override
  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    return ArgMatchResult(matched=True)


@experimental(FeatureName.TOOL_ARGS_MATCHER)
class ToolArgsMatcher(BaseModel):
  """Compares the arguments of a tool call against expected arguments.

  Only the arguments named in `expected` are compared, each with the matcher in
  `per_arg_matchers` for its name, or `default_matcher` otherwise. The call
  matches when every expected argument matches. An expected argument that is
  missing from the call never matches. To leave an argument unchecked, give it
  the `ignore` matcher or leave it out of `expected`.

  Example config, which loads with `ToolArgsMatcher.model_validate`:

  ```json
  {
    "perArgMatchers": {
      "city": {"type": "casefold"},
      "amount": {"type": "numeric", "tolerance": 0.01},
      "request_id": {"type": "ignore"}
    }
  }
  ```
  """

  model_config = _MATCHER_MODEL_CONFIG

  default_matcher: SerializeAsAny[BaseArgMatcher] = Field(
      default_factory=ExactArgMatcher
  )
  """The matcher for expected arguments not named in `per_arg_matchers`."""

  per_arg_matchers: dict[str, SerializeAsAny[BaseArgMatcher]] = Field(
      default_factory=dict
  )
  """Matchers keyed by argument name, overriding `default_matcher`."""

  allow_extra_args: bool = True
  """Whether the call may have arguments that `expected` does not name.

  Extra arguments affect only `matched`, not `score`, unless `match` is called
  with `short_circuit`, which may then skip comparisons.
  """

  @field_validator('default_matcher', mode='before')
  @classmethod
  def _parse_default_matcher(cls, value: Any) -> Any:
    return _to_arg_matcher(value)

  @field_validator('per_arg_matchers', mode='before')
  @classmethod
  def _parse_per_arg_matchers(cls, value: Any) -> Any:
    if not isinstance(value, Mapping):
      return value
    return {
        arg_name: _to_arg_matcher(matcher)
        for arg_name, matcher in value.items()
    }

  async def match(
      self,
      *,
      tool_name: str,
      expected: Mapping[str, object],
      actual: Mapping[str, object],
      short_circuit: bool = False,
  ) -> ToolArgsMatchResult:
    """Compares the actual arguments of a tool call with the expected ones.

    Arguments compared by deterministic matchers are compared first, one at a
    time. The remaining arguments are then compared concurrently. Exceptions
    raised by a matcher propagate to the caller, and the comparisons still
    running are cancelled.

    Args:
      tool_name: The name of the tool that was called.
      expected: The expected arguments, keyed by name.
      actual: The arguments the tool was called with, keyed by name.
      short_circuit: Whether to skip the non-deterministic comparisons once
        the deterministic ones or a disallowed extra argument decide
        `matched`. Use it when only `matched` is needed, for example to pick a
        rule; leave it off when the score or every per-argument result is
        needed. Skipped arguments are left out of `per_arg` and the score, so
        the score covers only the arguments that were compared and is 0.0
        when none were. A disallowed extra argument can therefore lower the
        score here, which it never does without `short_circuit`.

    Returns:
      The overall result and the result for each compared argument. With no
      expected arguments, the call matches unless it has extra arguments that
      are not allowed.
    """
    extra_args_ok = self.allow_extra_args or all(
        arg_name in expected for arg_name in actual
    )
    results: dict[str, ArgMatchResult] = {}
    deferred: list[tuple[str, BaseArgMatcher]] = []
    for arg_name, expected_value in expected.items():
      if arg_name not in actual:
        results[arg_name] = ArgMatchResult(
            matched=False, reason='The argument is missing from the call.'
        )
        continue
      matcher = self.per_arg_matchers.get(arg_name, self.default_matcher)
      if not matcher.is_deterministic:
        deferred.append((arg_name, matcher))
        continue
      results[arg_name] = await matcher.match(
          tool_name=tool_name,
          arg_name=arg_name,
          expected=expected_value,
          actual=actual[arg_name],
      )

    # A miss or a disallowed extra argument already decides `matched`.
    decided = not extra_args_ok or any(
        not result.matched for result in results.values()
    )
    if deferred and not (short_circuit and decided):
      tasks = [
          asyncio.create_task(
              matcher.match(
                  tool_name=tool_name,
                  arg_name=arg_name,
                  expected=expected[arg_name],
                  actual=actual[arg_name],
              )
          )
          for arg_name, matcher in deferred
      ]
      try:
        deferred_results = await asyncio.gather(*tasks)
      finally:
        # When one comparison raises, `gather` returns at once and leaves the
        # others running, which for a model-backed matcher means paying for
        # calls whose result nobody reads.
        for task in tasks:
          task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
      for (arg_name, _), result in zip(deferred, deferred_results):
        results[arg_name] = result

    per_arg = {
        arg_name: results[arg_name]
        for arg_name in expected
        if arg_name in results
    }
    matched = extra_args_ok and all(
        result.matched for result in per_arg.values()
    )
    if per_arg:
      score = sum(result.score for result in per_arg.values()) / len(per_arg)
    elif not expected:
      score = 1.0
    else:
      # short_circuit skipped every comparison because a disallowed extra
      # argument had already decided `matched`.
      score = 0.0
    return ToolArgsMatchResult(matched=matched, score=score, per_arg=per_arg)
