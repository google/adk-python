# Feature flags

ADK gates behavior that is not yet stable behind named feature flags, which you
turn on or off with the `ADK_ENABLE_<NAME>` and `ADK_DISABLE_<NAME>` environment
variables. You usually meet them because something warned at you about an
experimental feature, or refused to be constructed at all. `is_feature_enabled`
reads a flag, `override_feature_enabled` sets one from Python, and `FeatureName`
is the enumeration of the flags that exist.

## Introduction

You construct something, a `GCSToolset` for instance, and get a `UserWarning`
reading `[EXPERIMENTAL] feature <name> is enabled.` Or you disable something and
the next construction raises `RuntimeError: Feature <name> is not enabled.`

Both come from the feature registry. The warning says you are using a feature
that works but whose API may change; nothing is wrong. The `RuntimeError` says
the flag guarding that class is off, so the class refuses to be built at all.

The registry exists so that ADK can ship a feature to the people who want it
without changing behavior for everyone else. Each flag carries a default and one
of three lifecycle stages.

*   **Stable** features are on and silent.
*   **Experimental** features may be on or off depending on how far along they
    are, and they warn once per process when they run.
*   **Work in progress** features are off.

You never need this system for stable functionality. You need it when a release
note says a feature is behind a flag, when you want to suppress a change ADK
turned on by default, or when a construction call raises the `RuntimeError`
above.

## Get started

Turn a feature on for a whole process with an environment variable, before
starting it:

```shell
export ADK_ENABLE_SNAKE_CASE_SKILL_NAME=1
```

Or turn one off:

```shell
export ADK_DISABLE_JSON_SCHEMA_FOR_FUNC_DECL=1
```

The variable name is `ADK_ENABLE_` or `ADK_DISABLE_` followed by the flag's
name, which is the `FeatureName` member spelled exactly as it appears. Only the
values `1` and `true`, case-insensitive, count as set. Any other value, with
`yes`, `on` and `0` included, is treated as unset, which means the flag falls
through to the next rule rather than being forced off.

Where environment variables are awkward, such as a notebook or a test, set the
flag from Python instead. Do it before you construct anything that reads it:

```python
from google.adk.features import FeatureName
from google.adk.features import is_feature_enabled
from google.adk.features import override_feature_enabled

override_feature_enabled(FeatureName.SNAKE_CASE_SKILL_NAME, True)

assert is_feature_enabled(FeatureName.SNAKE_CASE_SKILL_NAME)
```

`FeatureName` is a `str` enum, so you can discover the current set with
`list(FeatureName)`. The membership changes between releases, so read it from
the version you have installed.

## How it works

`is_feature_enabled` resolves a flag in three steps and returns at the first one
that applies.

1.  **A programmatic override.** If `override_feature_enabled` has been called
    for this flag, that value wins outright.
2.  **Environment variables.** `ADK_ENABLE_<NAME>` is checked first and returns
    `True` if set to `1` or `true`. Then `ADK_DISABLE_<NAME>` is checked and
    returns `False` on the same values. Setting both means enable wins.
3.  **The registry default.** Each flag carries a stage and a default, and that
    default is the answer.

Two consequences follow from this ordering. `ADK_DISABLE_X=1` cannot switch off
a feature that a programmatic override has already turned on, so a library that
calls `override_feature_enabled` takes the decision away from whoever deploys
it. An override also cannot be removed through the public API, only flipped to
the other value, so calling `override_feature_enabled` in a test leaks into
every test that runs after it in the same process.

Whenever the resolution ends in "enabled" for a flag whose stage is not stable,
ADK emits a `UserWarning` whose text starts `[EXPERIMENTAL] feature` or
`[WIP] feature` and names the flag. It is emitted once per flag per process,
from `is_feature_enabled` itself, and it is purely informational, so nothing is
failing when you see it. Suppress it with the standard `warnings` filters if it
is noise in your logs.

Nothing is cached. Every call re-reads the override dictionary and
`os.environ`, so changing an environment variable mid-process does take effect.
Whether that helps depends on when the flag is read, and that varies by feature:
some are read on every call, some once when an object is constructed.

### What a flag actually gates

There are two enforcement styles, and the symptom differs.

**A gated unit.** Some classes and functions carry one of the `@experimental`,
`@working_in_progress` or `@stable` decorators, which call `is_feature_enabled`
before running. `GCSToolset`, `SpannerToolset`, `ComputerUseTool` and the
agent-config loaders are all gated this way. If the flag is off, constructing the
class or calling the function raises a `RuntimeError` saying the feature is not
enabled. The object is not half-built; the failure is immediate and total. Every
such flag defaults to on in the releases shipped so far, so you reach this error
only after deliberately disabling one.

**A gated code path.** Elsewhere the check sits inside a working feature and
selects between an old and a new behavior. Nothing raises. The flag changes
what happens, and the only way to know is the release note or the source.

## Functions and types

Three names cover the whole system: one to name a flag, one to read it, and one
to set it.

| Symbol | Signature | Description |
| :--- | :--- | :--- |
| `FeatureName` | `str` enum | The set of flags. Members change between releases. |
| `is_feature_enabled` | `(feature_name: FeatureName) -> bool` | Resolves a flag now, using the priority order above. |
| `override_feature_enabled` | `(feature_name: FeatureName, enabled: bool) -> None` | Sets the highest-priority override for the process. |

All three are exported from `google.adk.features`.

**`is_feature_enabled`** raises `ValueError` for a name that is not in the
registry. Since `FeatureName` members are always registered, that only happens
if you pass a bare string.

**`override_feature_enabled`** raises the same `ValueError` for an unregistered
name. It takes effect for the rest of the process and cannot be undone through
the public API.

## Advanced applications

Two situations need more than setting a variable and moving on: a test that must
not affect the tests after it, and a deployment that must behave the same way
after an upgrade.

### Scope a flag to a test

`override_feature_enabled` cannot be undone, so a test that flips a flag changes
every test that follows it in the same process. Set the environment variable
around the test instead, with pytest's `monkeypatch`, which restores the previous
value on teardown:

```python
def test_snake_case_skill_name(monkeypatch):
  monkeypatch.setenv("ADK_ENABLE_SNAKE_CASE_SKILL_NAME", "1")
  # Code under test reads the flag through is_feature_enabled.
```

The approach works because environment variables are read on every call rather
than cached, and because nothing in the test set a programmatic override, which
would have outranked it.

### Make a deployment reproducible

A default flipped in a later ADK release changes your agent's behavior on the
next deploy, and nothing in your own code changed to explain it. Set the flags
your agent depends on explicitly, in the deployment environment, rather than
relying on the registry default. Both directions are worth pinning:
`ADK_ENABLE_` for what you rely on, and `ADK_DISABLE_` for a default-on
experimental feature you have decided not to take yet.

## Limitations

*   **An override cannot be cleared.** The public API can set an override to
    `True` or `False`, but not remove it, so a process cannot return to
    environment-driven or default resolution once it has overridden a flag.
*   **`ADK_ENABLE_X=0` does not disable.** Only `1` and `true` are recognized as
    set, so a `0` reads as "not set" and resolution falls through to the
    registry default. Use `ADK_DISABLE_X=1` to actually turn something off.
*   **The flag set is not stable across releases.** Members are added and
    removed as features graduate, and a variable naming a flag that no longer
    exists is ignored, with no warning and no error.
*   **When a flag is read is feature-specific.** Some units read their flag at
    construction, some on every call. Setting a variable after the relevant
    object exists may have no effect.
*   **The warning has no per-flag switch.** Silencing the experimental warning
    means a `warnings` filter, which also hides other `UserWarning`s unless
    you match on the message.

## Related guides

*   Skill, Frontmatter, and Resources covers one
    concrete flag, `SNAKE_CASE_SKILL_NAME`, and what it changes.
