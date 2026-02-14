# feat(plugins): LlmResiliencePlugin – configurable retries/backoff and model fallbacks

## Motivation
Production agents need first-class resilience to transient LLM/API failures (timeouts, 429/5xx). Today, retry/fallback logic is ad-hoc and duplicated across projects. This PR introduces a plugin-based, opt-in resilience layer for LLM calls that aligns with ADK's extensibility philosophy and addresses recurring requests:

- #1214 Add built-in retry mechanism
- #2561 Retry mechanism gaps for common network errors (httpx…)
- Discussions: #2292, #3199 on fallbacks and max retries

## Summary
Adds a new plugin `LlmResiliencePlugin` which intercepts model errors and performs:
- Configurable retries with exponential backoff + jitter
- Transient error detection (HTTP 429/500/502/503/504, httpx timeouts/connect errors, asyncio timeouts)
- Optional model fallbacks (try a sequence of models if primary continues to fail)
- Works for standard `generate_content_async` flows; supports SSE streaming by consuming to final response

No core runner changes; this is a pure plugin. Default behavior remains unchanged unless the plugin is configured.

## Implementation Details
- File: `src/google/adk/plugins/llm_resilience_plugin.py`
- Hooks into `on_model_error_callback` to decide whether to handle an error
- Retries use exponential backoff with jitter (configurable):
  - `max_retries`, `backoff_initial`, `backoff_multiplier`, `max_backoff`, `jitter`
- Fallbacks use `LLMRegistry.new_llm(model)` to instantiate alternative models on failure
- Robust handling of provider return types:
  - Async generator (iterates until final non-partial response)
  - Coroutine (some providers may return a single `LlmResponse`)
- Avoids circular imports using duck-typed access to InvocationContext (works with Context alias)
- Maintains clean separation; no modification to runners or flows

## Tests
- `tests/unittests/plugins/test_llm_resilience_plugin.py`
  - `test_retry_success_on_same_model`: transient error triggers retry → success
  - `test_fallback_model_used_after_retries`: failing primary uses fallback model → success
  - `test_non_transient_error_bubbles`: non-transient error is ignored by plugin (propagate)

All tests in this module pass locally:

```
PYTHONPATH=src pytest -q tests/unittests/plugins/test_llm_resilience_plugin.py
# 3 passed
```

## Sample
- `samples/resilient_agent.py` demonstrates configuring the plugin with an in-memory runner and a demo model that fails once then succeeds.

Run sample:

```
PYTHONPATH=$(pwd)/src python samples/resilient_agent.py
```

## Backwards Compatibility
- Non-breaking: users opt-in by passing the plugin into `Runner(..., plugins=[LlmResiliencePlugin(...)])`
- No changes to public APIs beyond exporting the plugin in `google.adk.plugins`

## Limitations & Future Work
- Focused on LLM failures. Tool-level resilience is addressed by `ReflectAndRetryToolPlugin`.
- Circuit-breaking and per-exception policies could be added in a follow-up (`dev_3` item).
- Live bidi streaming not yet handled by this plugin; future work may extend to `BaseLlmConnection` flows.

## Docs
- Exported via `google.adk.plugins.__all__` to ease discovery
- Included inline docstrings and sample; can be integrated into the docs site in a separate PR

## Checklist
- [x] Unit tests for new behavior
- [x] Sample demonstrating usage
- [x] No changes to core runner/flow logic
- [x] Code formatted and linted per repository standards
