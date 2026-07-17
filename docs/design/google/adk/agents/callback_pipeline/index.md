# `_CallbackPipeline` - Code Unit Design

`_CallbackPipeline` centralizes ordered execution for agent-defined callbacks
without changing their existing public fields or runtime contracts. It is an
internal utility used by agent, model, and tool callback call sites.

## Introduction

ADK callback fields accept no callback, one callback, or a list containing a
mix of synchronous and asynchronous callbacks. Previously, each execution path
normalized and invoked these values independently, which made it easy for the
agent, model, async tool, and live tool paths to drift apart.

The implementation lives in
[`google.adk.agents._callback_pipeline`](https://github.com/google/adk-python/blob/main/src/google/adk/agents/_callback_pipeline.py).
Its behavior is covered by
[`test_callback_pipeline.py`](https://github.com/google/adk-python/blob/main/tests/unittests/agents/test_callback_pipeline.py)
and the existing agent, model, tool, live tool, and plugin callback integration
tests.

## High-level architecture

The unit contains four private building blocks:

- `_normalize_callbacks` converts `None` to an empty list, wraps a single
  callback, and returns an existing list unchanged to preserve list identity.
- `_CallbackPipeline` invokes callbacks in their configured order, awaits
  awaitable results, and forwards positional and keyword arguments unchanged.
- `_stop_on_truthy` preserves the stopping contract used by before and after
  agent, model, and tool callbacks.
- `_stop_on_non_none` preserves the stopping contract used by model-error and
  tool-error recovery callbacks, where an empty dictionary is a valid recovery
  response.

Each call site chooses its stop condition explicitly. If no callback satisfies
that condition, `execute` returns the final callback result; an empty pipeline
returns `None`. Callback exceptions are not intercepted and retain their
existing propagation behavior.

Plugin callbacks remain outside this unit. They use plugin-specific signatures
and retain their existing priority and fallback ordering before agent-defined
callbacks.

## Extension points

A new internal callback chain can reuse `_CallbackPipeline` when it has a
uniform callback signature and an explicitly defined stopping contract. The
generic parameter specification preserves the call signature, while the result
type connects callback results to the return type of `execute`.

New callback field shapes can reuse `_normalize_callbacks` when they support
the same `None`, single callback, and list forms.

## Extension constraints

- Select `_stop_on_truthy` or `_stop_on_non_none` from the callback field's
  established public behavior; do not infer the condition from the result type.
- Keep plugin callbacks outside the pipeline unless their public signature,
  priority, fallback, and exception contracts are intentionally redesigned.
- Do not expose the pipeline or its helpers from a public package module.
- Preserve callback order and allow exceptions to propagate unchanged.
- Do not copy callback lists during normalization, because callers may rely on
  identity and later list mutation.

## Limitations

The pipeline runs callbacks sequentially and does not provide parallel
execution, exception recovery, callback registration, or runtime mutation APIs.
It intentionally does not unify plugin-only hooks such as `on_agent_error`, and
it does not define a universal stopping condition for future callback types.
