# ToolArgsMatcher

`ToolArgsMatcher` decides whether the arguments of a tool call match the
arguments you expected, and how closely, one argument at a time. Each argument
can use its own comparison, from exact equality to a case-insensitive,
numeric-tolerance, or substring check, or a matcher you write and register.

## Introduction

Comparing a tool call's arguments with `==` is too strict for many agents. A
model that writes `"rome"` instead of `"Rome"`, passes `2.0` instead of `2`, or
adds a request ID that changes on every run fails an equality check even though
the call is correct. `ToolArgsMatcher` lets you say, per argument, what
"correct" means.

The module has two layers. A `BaseArgMatcher` compares one expected value with
one actual value and returns an `ArgMatchResult`. A `ToolArgsMatcher` applies
those matchers to every expected argument of a call and returns a
`ToolArgsMatchResult` that has both an overall `matched` decision and a `score`
between 0.0 and 1.0. Code that needs a yes or no answer, such as choosing which
recorded response to replay, reads `matched`. Code that needs partial credit,
such as an evaluation metric, reads `score`.

Matchers are Pydantic models, so a `ToolArgsMatcher` can be written in a JSON or
YAML config file as well as in Python. The API is experimental and may change
while the first callers adopt it.

## Get started

The following example compares a hotel search call with the arguments a test
expected. The city is compared without regard to case, the price is allowed to
differ by up to 5, and every other argument must be equal:

```python
from google.adk.tools.tool_args_matcher import ToolArgsMatcher

matcher = ToolArgsMatcher.model_validate({
    "perArgMatchers": {
        "city": {"type": "casefold"},
        "max_price": {"type": "numeric", "tolerance": 5},
    }
})

result = await matcher.match(
    tool_name="search_hotels",
    expected={"city": "Rome", "max_price": 200, "rooms": 1},
    actual={"city": "rome", "max_price": 203, "rooms": 2},
)

print(result.matched)  # False: rooms differs.
print(result.score)  # 0.666...: two of the three arguments matched.
print(result.per_arg["rooms"].matched)  # False
```

`match` is a coroutine because a matcher may call a model or a remote service.

## How it works

`ToolArgsMatcher.match` compares only the arguments named in `expected`. For
each one it picks the matcher configured for that argument name, or the default
matcher when none is configured, and records an `ArgMatchResult`. An expected
argument that is missing from the call never matches, and its matcher is not
called.

Arguments in the call that `expected` does not name are never compared, so they
never lower the score unless `short_circuit` is on (see below). Whether they make
the call fail to match is controlled by `allow_extra_args`.

The overall result combines the per-argument results:

* `matched` is true when every compared argument matched and the call has no
  disallowed extra arguments. To stop an argument from deciding the result,
  give it the `ignore` matcher or leave it out of `expected`.
* `score` is the mean of the per-argument scores. With nothing expected, the
  score is 1.0.
* `per_arg` holds each argument's result in the order of `expected`.

Matchers that always give the same answer for the same input run first, one at
a time. Matchers that do not, such as one that asks a model, run afterwards and
concurrently. With `short_circuit=True`, those slower matchers are skipped once
the faster results or a disallowed extra argument already decide `matched`,
which saves model calls when only the yes or no answer is needed. Skipped
arguments are absent from `per_arg` and from the score, which is 0.0 when
nothing was compared, so leave `short_circuit` off when you need the full
score.

If one of the concurrent matchers raises, the others still running are
cancelled before the exception reaches the caller.

An exception raised by a matcher propagates to the caller rather than counting
as a mismatch, because a silent mismatch can send a call down a path the caller
did not intend.

### Built-in matchers

Each built-in matcher is selected by its `type` name in config:

| `type` | Matches when | Notes |
| :--- | :--- | :--- |
| `exact` | The values are equal. | The default matcher. |
| `casefold` | Both strings are equal after trimming whitespace and ignoring case. | Values that are not both strings are compared for equality. |
| `numeric` | Both numbers differ by at most `tolerance`. | Booleans and non-numbers never match. `tolerance` defaults to 0.0. |
| `contains` | The expected string occurs inside the actual string. | Values that are not both strings are compared for equality. |
| `ignore` | Always, as long as the argument is present. | Useful for IDs, timestamps, and large payloads that change between runs. |

The names `exact`, `casefold`, `numeric`, `contains`, `ignore`, and `semantic`
are reserved for matchers that ship with ADK, and `register_arg_matcher`
rejects them for any other class.

## Configuration options

`ToolArgsMatcher` takes the following options. Config files use the camelCase
names, and Python code can use either form.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `default_matcher` | `BaseArgMatcher` | `exact` | The matcher for expected arguments not named in `per_arg_matchers`. |
| `per_arg_matchers` | `dict[str, BaseArgMatcher]` | `{}` | Matchers keyed by argument name, overriding `default_matcher`. |
| `allow_extra_args` | `bool` | `True` | Whether the call may have arguments that `expected` does not name. |

`default_matcher` sets the comparison for every argument at once. Changing it
from `exact` to `casefold`, for example, makes the whole call case-insensitive
while `per_arg_matchers` still overrides individual arguments.

`per_arg_matchers` is where most configuration goes, because different
arguments usually need different rules: an ID must be exact, a free-text query
may only need to contain a keyword, and a payload may not need checking at all.

`allow_extra_args` set to `False` makes a call with arguments beyond `expected`
fail to match, which reproduces a whole-dictionary equality check. Without
`short_circuit`, extra arguments still do not change `score`, so a caller that
only reads the score is unaffected by this option.

## Advanced applications

### Write a custom matcher

When none of the built-in rules fits, subclass `BaseArgMatcher`. Give the
`type` field a `Literal` type with a default, which names the matcher in config,
and implement `match`. Registering the class with `register_arg_matcher` makes
that name loadable from config. The following matcher accepts a date written
in several formats:

```python
from datetime import datetime
from typing import Literal

from google.adk.tools.tool_args_matcher import ArgMatchResult
from google.adk.tools.tool_args_matcher import BaseArgMatcher
from google.adk.tools.tool_args_matcher import register_arg_matcher

_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y")


def _parse(value: object) -> datetime | None:
  for fmt in _FORMATS:
    try:
      return datetime.strptime(str(value), fmt)
    except ValueError:
      continue
  return None


@register_arg_matcher
class DateArgMatcher(BaseArgMatcher):
  """Matches dates that are the same day in any supported format."""

  type: Literal["date"] = "date"

  async def match(
      self, *, tool_name: str, arg_name: str, expected: object, actual: object
  ) -> ArgMatchResult:
    expected_date, actual_date = _parse(expected), _parse(actual)
    if expected_date is None or actual_date is None:
      return ArgMatchResult(matched=False, reason="Not a recognized date.")
    return ArgMatchResult(matched=expected_date == actual_date)
```

Once the module that defines the matcher is imported, a config can name it as
`{"type": "date"}`. A matcher instance passed directly in Python works without
registration, because only config loading looks matchers up by name.

A matcher that calls a model or a remote service sets the class attribute
`is_deterministic = False`. `ToolArgsMatcher` then runs it after the
deterministic matchers, concurrently with other such matchers, and skips it
under `short_circuit` when the outcome is already known.

### Give partial credit for one argument

`ArgMatchResult.score` defaults to 1.0 for a match and 0.0 otherwise. A custom
matcher can set it explicitly, for example to a similarity value, and
`ToolArgsMatchResult.score` averages it with the other arguments. Keep `score`
between 0.0 and 1.0, which the result model enforces.

## Limitations

* Matchers compare top-level arguments. A nested value, such as one field
  inside a dictionary argument, is compared as a whole by the argument's
  matcher.
* Registration is process-wide. Two classes cannot register the same `type`
  name in one process, and registering the same class twice has no effect.
* A custom matcher must be imported before a config that names it is loaded,
  because the import is what registers it.
* The API is experimental and sits behind the `TOOL_ARGS_MATCHER` feature,
  which is on by default and logs an experimental warning once per process.
  With the feature turned off, for example by setting
  `ADK_DISABLE_TOOL_ARGS_MATCHER=1`, creating a `BaseArgMatcher` or
  `ToolArgsMatcher` raises `RuntimeError`.
