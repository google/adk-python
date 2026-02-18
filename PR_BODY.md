# feat(plugins): LlmResiliencePlugin – configurable retries/backoff and model fallbacks

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A
- Related: #1214
- Related: #2561
- Related discussions: #2292, #3199

**2. Or, if no issue exists, describe the change:**

**Problem:**
Production agents need first-class resilience to transient LLM/API failures
(timeouts, HTTP 429/5xx). Today, retry/fallback logic is often ad-hoc and
duplicated across projects.

**Solution:**
Introduce an opt-in plugin, `LlmResiliencePlugin`, that handles transient LLM
errors with configurable retries (exponential backoff + jitter) and optional
model fallbacks, without modifying core runner/flow logic.

### Summary

- Added `src/google/adk/plugins/llm_resilience_plugin.py`.
- Exported `LlmResiliencePlugin` in `src/google/adk/plugins/__init__.py`.
- Added unit tests in
  `tests/unittests/plugins/test_llm_resilience_plugin.py`:
  - `test_retry_success_on_same_model`
  - `test_fallback_model_used_after_retries`
  - `test_non_transient_error_bubbles`
- Added `samples/resilient_agent.py` demo.

### Testing Plan

**Unit Tests:**

- [x] I have added or updated unit tests for my change.
- [x] All unit tests pass locally.

Command run:

```shell
.venv/Scripts/python -m pytest tests/unittests/plugins/test_llm_resilience_plugin.py -v
```

Result summary:

```text
collected 3 items
tests/unittests/plugins/test_llm_resilience_plugin.py::TestLlmResiliencePlugin::test_fallback_model_used_after_retries PASSED
tests/unittests/plugins/test_llm_resilience_plugin.py::TestLlmResiliencePlugin::test_non_transient_error_bubbles PASSED
tests/unittests/plugins/test_llm_resilience_plugin.py::TestLlmResiliencePlugin::test_retry_success_on_same_model PASSED
3 passed
```

**Manual End-to-End (E2E) Tests:**

Run sample:

```shell
.venv/Scripts/python samples/resilient_agent.py
```

Observed output:

```text
LLM retry attempt 1 failed: TimeoutError('Simulated transient failure')
Collected 1 events
MODEL: Recovered on retry!
```

### Checklist

- [x] I have read the [CONTRIBUTING.md](https://github.com/google/adk-python/blob/main/CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
- [x] Any dependent changes have been merged and published in downstream modules. (N/A; no dependent changes)

### Additional context

- Non-breaking: users opt in via
  `Runner(..., plugins=[LlmResiliencePlugin(...)])`.
- Transient detection currently targets common HTTP/timeouts and can be extended
  in follow-ups (e.g., per-exception policy, circuit breaking).
- Live bidirectional streaming paths are out of scope for this PR.
