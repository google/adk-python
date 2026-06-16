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

"""Tests for per-request telemetry configuration overrides."""

from __future__ import annotations

import asyncio
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry import ContentCapturingMode
from google.adk.telemetry import TelemetryConfig
from google.adk.telemetry import tracing
from google.adk.telemetry._experimental_semconv import set_operation_details_common_attributes
from google.adk.telemetry.context import ADK_TELEMETRY_IGNORE_RUN_CONFIG
from google.adk.telemetry.tracing import trace_inference_result
from google.genai.types import Part
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError
import pytest

from ..testing_utils import InMemoryRunner
from ..testing_utils import MockModel
from ..testing_utils import UserContent

_ENV_EXPERIMENTAL = 'OTEL_SEMCONV_STABILITY_OPT_IN'
_ENV_CAPTURE = 'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'
_ENV_ADK_SPAN_CAPTURE = 'ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS'
_ENV_ADMIN_LOCK = ADK_TELEMETRY_IGNORE_RUN_CONFIG

_ALL_TELEMETRY_ENV_VARS = (
    _ENV_EXPERIMENTAL,
    _ENV_CAPTURE,
    _ENV_ADK_SPAN_CAPTURE,
    _ENV_ADMIN_LOCK,
)


def _set_env(monkeypatch: pytest.MonkeyPatch, **env: Optional[str]) -> None:
  """Applies a clean telemetry env: unset everything, then set the given vars.

  Starting from a known-empty state keeps each parametrized case hermetic
  regardless of what the host environment happens to export.
  """
  for name in _ALL_TELEMETRY_ENV_VARS:
    monkeypatch.delenv(name, raising=False)
  for name, value in env.items():
    if value is not None:
      monkeypatch.setenv(name, value)


def test_telemetry_config_is_frozen():
  """Frozen TelemetryConfig rejects mutation after construction."""
  cfg = TelemetryConfig(genai_semconv_stability_opt_in='experimental')
  with pytest.raises(ValidationError):
    cfg.genai_semconv_stability_opt_in = 'stable'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Construction truth table for ``TelemetryConfig`` itself (no env vars, no
# decision functions). Covers the cartesian product of the two fields'
# accepted/rejected values: every valid combination must construct and
# preserve its field values; every invalid value must raise ValidationError.
# ---------------------------------------------------------------------------

# Accepted values for each field. ``None`` means "field left at default".
_VALID_OPT_IN_VALUES = (None, 'stable', 'experimental')
_VALID_CAPTURE_VALUES = (
    None,
    ContentCapturingMode.NO_CONTENT,
    ContentCapturingMode.EVENT_ONLY,
    ContentCapturingMode.SPAN_ONLY,
    ContentCapturingMode.SPAN_AND_EVENT,
)

# Full cartesian product of valid field values.
_VALID_CONSTRUCTION_TABLE = [
    (opt_in, capture)
    for opt_in in _VALID_OPT_IN_VALUES
    for capture in _VALID_CAPTURE_VALUES
]


@pytest.mark.parametrize('opt_in,capture', _VALID_CONSTRUCTION_TABLE)
def test_telemetry_config_construction_accepts_valid_combinations(
    opt_in: Optional[str],
    capture: Optional[ContentCapturingMode],
):
  """Every valid (opt_in, capture) pair constructs and round-trips its fields."""
  cfg = TelemetryConfig(
      genai_semconv_stability_opt_in=opt_in,
      capture_message_content=capture,
  )
  assert cfg.genai_semconv_stability_opt_in == opt_in
  assert cfg.capture_message_content == capture


@pytest.mark.parametrize('member', list(ContentCapturingMode))
def test_telemetry_config_construction_coerces_capture_member_value_str(
    member: ContentCapturingMode,
):
  """A ``ContentCapturingMode`` value string coerces to the enum member.

  pydantic accepts the enum member's underlying value (e.g. ``'EVENT_ONLY'``)
  and coerces it to the member, mirroring the env-var path which accepts the
  matching uppercase string. Pinned so this leniency stays intentional.
  """
  cfg = TelemetryConfig(capture_message_content=member.value)  # type: ignore[arg-type]
  assert cfg.capture_message_content is member


# Each entry is (kwargs, reason); constructing with the kwargs must raise.
_INVALID_CONSTRUCTION_TABLE = [
    # genai_semconv_stability_opt_in only accepts the two literals.
    (
        {'genai_semconv_stability_opt_in': 'gen_ai_latest_experimental'},
        'opt_in',
    ),
    ({'genai_semconv_stability_opt_in': 'STABLE'}, 'opt_in_case'),
    ({'genai_semconv_stability_opt_in': ''}, 'opt_in_empty'),
    # capture_message_content must be a valid ContentCapturingMode (member or
    # its value string); strings outside that set are rejected.
    ({'capture_message_content': 'not_a_capture_mode'}, 'capture_invalid'),
    ({'capture_message_content': 'event_only'}, 'capture_wrong_case'),
    # extra='forbid' rejects unknown fields.
    ({'typo_field': 'experimental'}, 'extra_field'),
]


@pytest.mark.parametrize(
    'kwargs,_reason',
    _INVALID_CONSTRUCTION_TABLE,
    ids=[reason for _, reason in _INVALID_CONSTRUCTION_TABLE],
)
def test_telemetry_config_construction_rejects_invalid_values(
    kwargs: dict,
    _reason: str,
):
  """Invalid field values / unknown fields raise ValidationError on construct."""
  with pytest.raises(ValidationError):
    TelemetryConfig(**kwargs)  # type: ignore[arg-type]


def test_telemetry_config_round_trips_through_json():
  """``RunConfig.telemetry`` round-trips through JSON."""
  cfg = RunConfig(
      telemetry=TelemetryConfig(
          genai_semconv_stability_opt_in='experimental',
          capture_message_content=ContentCapturingMode.SPAN_AND_EVENT,
      )
  )
  js = cfg.model_dump_json()
  assert 'experimental' in js
  assert 'SPAN_AND_EVENT' in js
  reloaded = RunConfig.model_validate_json(js)
  assert reloaded.telemetry == cfg.telemetry
  assert isinstance(reloaded.telemetry, TelemetryConfig)
  assert isinstance(
      reloaded.telemetry.capture_message_content, ContentCapturingMode
  )


# ---------------------------------------------------------------------------
# Env-string parsing edge cases for the capture-mode env var.
#
# The precedence ladder (admin lock > cfg > env > default) for all four
# decision functions is verified end-to-end by the functional ``Runner``
# tests below; these unit tests pin only the env-string parsing quirks that
# the functional tests cannot exercise directly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('invalid', ['yes', 'on', 'not_a_capture_mode'])
def test_capture_mode_env_invalid_values_treated_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
):
  """Env values outside the OTel four-state set fall back to ''.

  Legacy ``'true'`` / ``'1'`` are the only exception; see
  ``test_capture_mode_env_legacy_*``.
  """
  monkeypatch.setenv(_ENV_CAPTURE, invalid)
  assert TelemetryConfig().content_capturing_mode_value == ''


@pytest.mark.parametrize('legacy', ['true', 'TRUE', 'True', '1'])
def test_capture_mode_env_legacy_values_coerced_to_event_only(
    monkeypatch: pytest.MonkeyPatch,
    legacy: str,
):
  """Legacy ``'true'`` / ``'1'`` coerce silently to ``EVENT_ONLY``.

  Previously the env path was bool-ish; the canonical value space is now
  the OTel four-state enum (matching
  ``opentelemetry.util.genai.utils.get_content_capturing_mode``).
  Coercion preserves observable behavior for existing deployments.
  """
  monkeypatch.setenv(_ENV_CAPTURE, legacy)
  assert TelemetryConfig().content_capturing_mode_value == 'EVENT_ONLY', (
      f"legacy env value {legacy!r} should coerce to 'EVENT_ONLY' for"
      ' back-compat'
  )


@pytest.mark.parametrize('legacy', ['true', '1'])
def test_capture_mode_env_legacy_coercion_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    legacy: str,
):
  """Legacy coercion is silent (hot path; no per-span log records)."""
  monkeypatch.setenv(_ENV_CAPTURE, legacy)
  with caplog.at_level(
      'WARNING', logger='google.adk.telemetry._experimental_semconv'
  ):
    assert TelemetryConfig().content_capturing_mode_value == 'EVENT_ONLY'
  assert not caplog.records, (
      'legacy-value coercion must be silent; got log records:'
      f' {[(r.levelname, r.message) for r in caplog.records]}'
  )


# ---------------------------------------------------------------------------
# TelemetryConfig resolution properties: the single source of truth.
#
# The decision functions are now thin wrappers over these properties, so the
# precedence ladder (admin lock > per-request field > env var > default) and
# the env-string coercion are pinned here once, directly on the model. The
# functional Runner tests above exercise the same ladder end-to-end; these are
# the fast, exhaustive unit-level guards.
# ---------------------------------------------------------------------------


def test_resolution_properties_read_env_lazily_at_access_time(
    monkeypatch: pytest.MonkeyPatch,
):
  """Properties re-read os.environ on each access (not frozen at construction).

  This is the whole reason resolution lives in properties rather than a
  ``default_factory``: a caller that mutates the environment after building the
  (frozen) config still observes the new value.
  """
  _set_env(monkeypatch)
  cfg = TelemetryConfig()  # all fields unset => pure env-fallback.
  assert cfg.should_use_experimental_genai_semconv is False
  monkeypatch.setenv(_ENV_EXPERIMENTAL, 'gen_ai_latest_experimental')
  assert cfg.should_use_experimental_genai_semconv is True
  monkeypatch.delenv(_ENV_EXPERIMENTAL, raising=False)
  assert cfg.should_use_experimental_genai_semconv is False


# (opt_in field, env value, expected) for should_use_experimental_genai_semconv.
# Covers field-wins, env-fallback, and the 'stable' field beating an opted-in
# env var.
_EXPERIMENTAL_RESOLUTION_TABLE = [
    # Field set => field wins over env.
    ('experimental', None, True),
    ('experimental', 'gen_ai_latest_experimental', True),
    ('stable', 'gen_ai_latest_experimental', False),
    ('stable', None, False),
    # Field unset => env fallback (token must appear in the CSV list).
    (None, 'gen_ai_latest_experimental', True),
    (None, 'gen_ai_latest_experimental,http_latest', True),
    (None, 'http_latest', False),
    (None, None, False),
]


@pytest.mark.parametrize(
    'opt_in,env_value,expected', _EXPERIMENTAL_RESOLUTION_TABLE
)
def test_should_use_experimental_genai_semconv_resolution(
    monkeypatch: pytest.MonkeyPatch,
    opt_in: Optional[str],
    env_value: Optional[str],
    expected: bool,
):
  """Per-request opt_in wins; otherwise the env CSV opt-in token decides."""
  _set_env(monkeypatch, **{_ENV_EXPERIMENTAL: env_value})
  cfg = TelemetryConfig(genai_semconv_stability_opt_in=opt_in)
  assert cfg.should_use_experimental_genai_semconv is expected


# (capture field, env value, expected mode value string). Exercises field-wins,
# env fallback, the legacy 'true'/'1' coercion, and invalid-env => NO_CONTENT.
_CAPTURE_RESOLUTION_TABLE = [
    # Field set => field wins over env (NO_CONTENT maps to '').
    (ContentCapturingMode.NO_CONTENT, 'SPAN_AND_EVENT', ''),
    (ContentCapturingMode.EVENT_ONLY, None, 'EVENT_ONLY'),
    (ContentCapturingMode.SPAN_ONLY, 'EVENT_ONLY', 'SPAN_ONLY'),
    (ContentCapturingMode.SPAN_AND_EVENT, None, 'SPAN_AND_EVENT'),
    # Field unset => env fallback over the OTel four-state set.
    (None, 'EVENT_ONLY', 'EVENT_ONLY'),
    (None, 'SPAN_AND_EVENT', 'SPAN_AND_EVENT'),
    (None, 'NO_CONTENT', ''),
    # Field unset => legacy back-compat coercion of 'true'/'1' to EVENT_ONLY.
    (None, 'true', 'EVENT_ONLY'),
    (None, '1', 'EVENT_ONLY'),
    # Field unset => invalid / absent env value => NO_CONTENT ('').
    (None, 'bogus', ''),
    (None, None, ''),
]


@pytest.mark.parametrize(
    'capture,env_value,expected', _CAPTURE_RESOLUTION_TABLE
)
def test_content_capturing_mode_value_resolution(
    monkeypatch: pytest.MonkeyPatch,
    capture: Optional[ContentCapturingMode],
    env_value: Optional[str],
    expected: str,
):
  """Per-request capture field wins; else env fallback w/ legacy coercion."""
  _set_env(monkeypatch, **{_ENV_CAPTURE: env_value})
  cfg = TelemetryConfig(capture_message_content=capture)
  assert cfg.content_capturing_mode_value == expected


# (resolved mode, expect_logs, expect_experimental_spans). Pins the OTel-spec
# routing of a resolved mode onto the LogRecord vs span side.
_CONTENT_ROUTING_TABLE = [
    (ContentCapturingMode.NO_CONTENT, False, False),
    (ContentCapturingMode.EVENT_ONLY, True, False),
    (ContentCapturingMode.SPAN_ONLY, False, True),
    (ContentCapturingMode.SPAN_AND_EVENT, True, True),
]


@pytest.mark.parametrize(
    'mode,expect_logs,expect_spans', _CONTENT_ROUTING_TABLE
)
def test_content_routing_logs_vs_experimental_spans(
    monkeypatch: pytest.MonkeyPatch,
    mode: ContentCapturingMode,
    expect_logs: bool,
    expect_spans: bool,
):
  """EVENT_ONLY routes to logs, SPAN_ONLY to spans, SPAN_AND_EVENT to both."""
  _set_env(monkeypatch)
  cfg = TelemetryConfig(capture_message_content=mode)
  assert cfg.should_add_content_to_logs is expect_logs
  assert cfg.should_add_content_to_experimental_spans is expect_spans


# (capture field, ADK span env value, expected). The legacy ADK span knob has
# its OWN env fallback (ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, default-on),
# distinct from the OTel content env var.
_LEGACY_SPAN_RESOLUTION_TABLE = [
    # Field set => OTel-spec routing (only span-bearing modes opt in).
    (ContentCapturingMode.SPAN_ONLY, 'false', True),
    (ContentCapturingMode.SPAN_AND_EVENT, 'false', True),
    (ContentCapturingMode.EVENT_ONLY, 'true', False),
    (ContentCapturingMode.NO_CONTENT, 'true', False),
    # Field unset => ADK span env var, which DEFAULTS TO ON.
    (None, None, True),
    (None, 'true', True),
    (None, '1', True),
    (None, 'false', False),
    (None, '0', False),
]


@pytest.mark.parametrize(
    'capture,env_value,expected', _LEGACY_SPAN_RESOLUTION_TABLE
)
def test_should_add_content_to_legacy_spans_resolution(
    monkeypatch: pytest.MonkeyPatch,
    capture: Optional[ContentCapturingMode],
    env_value: Optional[str],
    expected: bool,
):
  """Legacy ADK span knob: field uses OTel routing, else its own default-on env."""
  _set_env(monkeypatch, **{_ENV_ADK_SPAN_CAPTURE: env_value})
  cfg = TelemetryConfig(capture_message_content=capture)
  assert cfg.should_add_content_to_legacy_spans is expected


def test_admin_lock_disables_all_resolution_properties(
    monkeypatch: pytest.MonkeyPatch,
):
  """Admin lock makes every resolution property ignore the per-request fields.

  With the lock on and all env vars unset, an opted-in config resolves to the
  (empty) env defaults across all properties. The legacy span knob still
  defaults to on (its env default), matching pre-CL behavior.
  """
  _set_env(monkeypatch, **{_ENV_ADMIN_LOCK: '1'})
  cfg = TelemetryConfig(
      genai_semconv_stability_opt_in='experimental',
      capture_message_content=ContentCapturingMode.SPAN_AND_EVENT,
  )
  assert cfg.should_use_experimental_genai_semconv is False
  assert cfg.content_capturing_mode_value == ''
  assert cfg.should_add_content_to_logs is False
  assert cfg.should_add_content_to_experimental_spans is False
  # Legacy span knob falls back to its env var, which defaults to on.
  assert cfg.should_add_content_to_legacy_spans is True


def test_admin_lock_falls_back_to_env_not_per_request_field(
    monkeypatch: pytest.MonkeyPatch,
):
  """With the lock on, env wins over the ignored per-request field.

  Complements ``test_admin_lock_disables_all_resolution_properties``, which runs
  with env unset and so only proves the lock falls back to the env *default*.
  Here the per-request fields are set to *suppressing* values while the env vars
  are set to *capturing* values, proving each property reads the env (the
  operator's setting) rather than forcing the value off.
  """
  _set_env(
      monkeypatch,
      **{
          _ENV_ADMIN_LOCK: '1',
          _ENV_EXPERIMENTAL: 'gen_ai_latest_experimental',
          _ENV_CAPTURE: 'EVENT_ONLY',
          _ENV_ADK_SPAN_CAPTURE: 'false',
      },
  )
  cfg = TelemetryConfig(
      genai_semconv_stability_opt_in='stable',
      capture_message_content=ContentCapturingMode.NO_CONTENT,
  )
  # Env opts in even though the per-request field said 'stable'.
  assert cfg.should_use_experimental_genai_semconv is True
  # Env capturing mode wins even though the field said NO_CONTENT.
  assert cfg.content_capturing_mode_value == 'EVENT_ONLY'
  assert cfg.should_add_content_to_logs is True
  # Env says EVENT_ONLY (not span-bearing), so experimental spans stay off.
  assert cfg.should_add_content_to_experimental_spans is False
  # Legacy span env explicitly set to false wins over the ignored field.
  assert cfg.should_add_content_to_legacy_spans is False


# ---------------------------------------------------------------------------
# set_operation_details_common_attributes: must honor the per-request config.
#
# log_only_attributes carry PII-ish data (e.g. user_id). The gate must consult
# the per-request TelemetryConfig, not just the process-global env var, or a
# request that opted out via capture_message_content=NO_CONTENT would still leak
# log-only attributes when the host env defaults to a content-capturing mode.
# ---------------------------------------------------------------------------


def _run_set_common_attrs(
    telemetry_config: Optional[TelemetryConfig],
) -> dict:
  """Runs set_operation_details_common_attributes and returns the result map."""
  out: dict = {}
  set_operation_details_common_attributes(
      out,
      telemetry_config or TelemetryConfig(),
      {'gen_ai.operation.name': 'chat'},
      log_only_attributes={'gen_ai.user.id': 'user-123'},
  )
  return out


def test_set_common_attrs_cfg_no_content_overrides_env_capture(
    monkeypatch: pytest.MonkeyPatch,
):
  """Per-request NO_CONTENT suppresses PII-ish log-only attrs even if env opts in.

  Security regression guard: ``log_only_attributes`` (e.g. ``user_id``) must be
  gated on the per-request config, not just the process-global env var, or a
  request that opted out via ``capture_message_content=NO_CONTENT`` would leak
  log-only attributes when the host env defaults to a content-capturing mode.
  The functional ``Runner`` tests do not assert on log-only attribute routing,
  so this stays as a dedicated unit guard.
  """
  monkeypatch.setenv(_ENV_CAPTURE, 'EVENT_ONLY')
  out = _run_set_common_attrs(
      TelemetryConfig(capture_message_content=ContentCapturingMode.NO_CONTENT)
  )
  assert 'gen_ai.user.id' not in out
  # Non-log-only attributes are always set.
  assert out['gen_ai.operation.name'] == 'chat'


# ---------------------------------------------------------------------------
# trace_inference_result: invocation_context is Optional[InvocationContext].
# TelemetryContext.invocation_context is Optional, so the signature must accept
# None without raising (None-safe via _telemetry_config_from_invocation_context).
# ---------------------------------------------------------------------------


def test_trace_inference_result_accepts_none_invocation_context():
  """Passing invocation_context=None must not raise (env fallback path)."""
  # span=None short-circuits inside the function; the point is that a None
  # invocation_context does not blow up signature/None handling.
  trace_inference_result(None, None, LlmResponse())


# ---------------------------------------------------------------------------
# Admin-lock value parsing: which env-var spellings count as "locked".
#
# The lock's *effect* (ignoring per-request cfg, falling back to env) is
# verified end-to-end by the admin-lock functional ``Runner`` tests below.
# What those cannot vary is the lock env-var *spelling*, so this single
# parametrized test pins the truthy/falsy matrix once, asserting across all
# four decision functions at the same time.
# ---------------------------------------------------------------------------

# (lock_value, locked) -- 'locked' True means per-request cfg is ignored.
_ADMIN_LOCK_VALUE_TABLE = [
    # Recognized truthy spellings (case-insensitive '1' / 'true').
    ('1', True),
    ('true', True),
    ('TRUE', True),
    ('True', True),
    # The parser strips surrounding whitespace before comparing.
    (' 1 ', True),
    # Everything else is treated as unset: explicit off-ish spellings and
    # arbitrary unrecognized strings (the lock uses a strict allowlist, not a
    # generic truthy parse, so 'yes' does NOT lock).
    ('', False),
    ('0', False),
    ('false', False),
    ('no', False),
    ('off', False),
    ('yes', False),
]


@pytest.mark.parametrize('lock_value,locked', _ADMIN_LOCK_VALUE_TABLE)
def test_admin_lock_value_parsing(
    monkeypatch: pytest.MonkeyPatch,
    lock_value: str,
    locked: bool,
):
  """Only stripped '1'/'true' (case-insensitive) lock; all else is unset.

  When locked, a per-request cfg opting in to experimental + EVENT_ONLY is
  ignored and the (empty) env fallback wins; when unlocked, the cfg wins.
  Asserts across all four decision functions to pin the shared parsing.
  """
  _set_env(monkeypatch, **{_ENV_ADMIN_LOCK: lock_value})
  cfg = TelemetryConfig(
      genai_semconv_stability_opt_in='experimental',
      capture_message_content=ContentCapturingMode.EVENT_ONLY,
  )
  assert cfg.should_use_experimental_genai_semconv is (not locked)
  assert cfg.should_add_content_to_logs is (not locked)
  assert bool(cfg.content_capturing_mode_value) is (not locked)
  # SPAN-bearing knob: EVENT_ONLY does not enable spans, so when unlocked the
  # cfg disables span capture; when locked the env default (on) wins.
  assert cfg.should_add_content_to_legacy_spans is locked


def _make_test_runner(
    agent_name: str = 'telemetry_test_agent',
) -> InMemoryRunner:
  """Builds an InMemoryRunner with a deterministic MockModel.

  Each invocation drives one ``generate_content`` call returning a short
  text response, which is enough to exercise the ``use_inference_span`` +
  ``trace_inference_result`` code path that consumes ``RunConfig.telemetry``.
  """
  mock_model = MockModel.create(
      responses=[Part.from_text(text='ok')],
  )
  agent = Agent(name=agent_name, model=mock_model)
  return InMemoryRunner(root_agent=agent)


@pytest.fixture
def span_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> InMemorySpanExporter:
  """Captures every span emitted by ``tracing.tracer`` during a test.

  Mirrors the fixture in ``telemetry/test_functional.py`` so that any
  future change to how the tracer is wired up is picked up here too.
  """
  tracer_provider = TracerProvider()
  exporter = InMemorySpanExporter()
  tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
  real_tracer = tracer_provider.get_tracer(__name__)
  monkeypatch.setattr(
      tracing.tracer,
      'start_as_current_span',
      real_tracer.start_as_current_span,
  )
  return exporter


# LogRecord event names that distinguish the experimental-semconv path from
# the stable-semconv path. These are the actual emission contract the CL
# delivers, so asserting on their presence (rather than on
# implementation-detail span attributes) gives a robust signal: either the
# experimental code path ran (its LogRecord appears) or the stable code path
# ran (its three LogRecords appear). The two sets are mutually exclusive at
# the call site -- see ``use_inference_span`` (tracing.py) +
# ``maybe_log_completion_details`` (_experimental_semconv.py).
_EXPERIMENTAL_LOG_EVENT = 'gen_ai.client.inference.operation.details'
_STABLE_LOG_EVENTS = frozenset({
    'gen_ai.system.message',
    'gen_ai.user.message',
    'gen_ai.choice',
})


@pytest.fixture
def log_collector(monkeypatch: pytest.MonkeyPatch) -> list:
  """Captures every LogRecord emitted by ``tracing.otel_logger`` during a test.

  Patches the module-global rather than wiring an OTel ``LoggerProvider``,
  matching how ``test_spans.py`` drives ``otel_logger`` assertions. The
  returned list is mutated in place by the patched ``emit``, so assertions
  can run after the Runner finishes without copying.

  Returns:
    A list of the OTel ``LogRecord``s emitted during the test.
  """
  collected: list = []
  real_emit = tracing.otel_logger.emit

  def _collecting_emit(record, *args, **kwargs):
    collected.append(record)
    return real_emit(record, *args, **kwargs)

  monkeypatch.setattr(tracing.otel_logger, 'emit', _collecting_emit)
  return collected


def _experimental_logs(records: list) -> list:
  """Filters captured LogRecords down to the experimental-semconv emission."""
  return [r for r in records if r.event_name == _EXPERIMENTAL_LOG_EVENT]


def _stable_logs(records: list) -> list:
  """Filters captured LogRecords down to stable-semconv emissions."""
  return [r for r in records if r.event_name in _STABLE_LOG_EVENTS]


@pytest.mark.asyncio
async def test_runner_invocation_with_experimental_telemetry_emits_experimental_log(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_collector: list,
):
  """Per-request experimental opt-in drives the experimental emission end-to-end.

  Runs one invocation with
  ``genai_semconv_stability_opt_in='experimental'`` and asserts a
  ``gen_ai.client.inference.operation.details`` LogRecord was emitted.
  ``maybe_log_completion_details`` emits it only when
  ``is_experimental_semconv(telemetry_config)`` is True, so its presence
  proves the config was threaded through ``Runner.run_async`` ->
  ``InvocationContext.run_config.telemetry`` -> ``use_inference_span`` ->
  ``is_experimental_semconv``.

  ``span_exporter`` is wired in because ``maybe_log_completion_details``
  bails out when its ``span`` argument is ``None``.
  """
  monkeypatch.delenv(_ENV_EXPERIMENTAL, raising=False)
  runner = _make_test_runner()
  session = await runner.runner.session_service.create_session(
      app_name=runner.app_name, user_id='test_user'
  )
  async for _ in runner.runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=UserContent('hi'),
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              genai_semconv_stability_opt_in='experimental'
          )
      ),
  ):
    pass

  exp = _experimental_logs(log_collector)
  stable = _stable_logs(log_collector)
  assert exp, (
      f'expected at least one {_EXPERIMENTAL_LOG_EVENT!r} LogRecord but got'
      ' none; emitted event_names='
      f'{sorted({r.event_name for r in log_collector})}.  This means'
      ' RunConfig.telemetry was NOT threaded into use_inference_span and the'
      ' helper fell back to the stable-semconv code path.'
  )
  assert not stable, (
      'experimental path should NOT emit the stable-semconv per-message'
      f' LogRecords ({sorted(_STABLE_LOG_EVENTS)}); got'
      f' {sorted({r.event_name for r in stable})}.'
  )


@pytest.mark.asyncio
async def test_runner_invocation_without_telemetry_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_collector: list,
):
  """No per-request config => env var path remains in effect (back-compat).

  Prior behavior must be preserved when ``RunConfig.telemetry`` is unset:
  the env var ``OTEL_SEMCONV_STABILITY_OPT_IN`` should still flip the
  experimental path on, so the experimental LogRecord still appears.
  """
  monkeypatch.setenv(_ENV_EXPERIMENTAL, 'gen_ai_latest_experimental')
  runner = _make_test_runner()
  session = await runner.runner.session_service.create_session(
      app_name=runner.app_name, user_id='test_user'
  )
  async for _ in runner.runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=UserContent('hi'),
      # No telemetry= field => RunConfig.telemetry defaults to None.
      run_config=RunConfig(),
  ):
    pass

  exp = _experimental_logs(log_collector)
  assert exp, (
      'env-var path no longer enables experimental semconv when'
      ' RunConfig.telemetry is None -- regression on backward compat.'
      f'  emitted event_names={sorted({r.event_name for r in log_collector})}'
  )


@pytest.mark.asyncio
async def test_runner_invocation_with_experimental_false_takes_stable_path(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_collector: list,
):
  """Per-request 'stable' suppresses the experimental emission even when env opts in.

  An invocation that has not opted into the experimental schema keeps
  getting stable-semconv emissions even while a process-global env var
  (set by the host for other invocations) requests the experimental
  schema.
  """
  monkeypatch.setenv(_ENV_EXPERIMENTAL, 'gen_ai_latest_experimental')
  runner = _make_test_runner()
  session = await runner.runner.session_service.create_session(
      app_name=runner.app_name, user_id='test_user'
  )
  async for _ in runner.runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=UserContent('hi'),
      run_config=RunConfig(
          telemetry=TelemetryConfig(genai_semconv_stability_opt_in='stable')
      ),
  ):
    pass

  exp = _experimental_logs(log_collector)
  stable = _stable_logs(log_collector)
  assert not exp, (
      "genai_semconv_stability_opt_in='stable' should suppress the"
      f' {_EXPERIMENTAL_LOG_EVENT!r} LogRecord regardless of the env var,'
      f' but got {len(exp)} such records. The env-var fallback beat the'
      ' explicit per-request override.'
  )
  assert stable, (
      'stable-semconv path should emit per-message LogRecords'
      f' ({sorted(_STABLE_LOG_EVENTS)}); got none.  emitted event_names='
      f'{sorted({r.event_name for r in log_collector})}'
  )


@pytest.mark.asyncio
async def test_concurrent_runner_invocations_do_not_leak_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_collector: list,
):
  """Two concurrent invocations with opposite TelemetryConfigs each see their own.

  Runs two ``InMemoryRunner`` instances (one per simulated tenant)
  concurrently via ``asyncio.gather`` and asserts each tenant's emitted
  LogRecords match its own config, so ``telemetry_config`` did not leak
  across the two requests.

  Tenants are disambiguated by ``gen_ai.agent.name`` on the experimental
  record's attributes; stable per-message records carry no agent identity,
  so the opted-out tenant is identified by the absence of its experimental
  record plus the presence of stable per-message records overall.

  Isolation holds by construction (no shared mutable state); pinned so a
  future global or contextvar inside the decision functions trips this
  test.
  """
  monkeypatch.delenv(_ENV_EXPERIMENTAL, raising=False)

  tenant_on = _make_test_runner('tenant_on')
  tenant_off = _make_test_runner('tenant_off')

  async def _run(
      runner: InMemoryRunner,
      telemetry_config: Optional[TelemetryConfig],
  ) -> None:
    session = await runner.runner.session_service.create_session(
        app_name=runner.app_name, user_id='test_user'
    )
    async for _ in runner.runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=UserContent('hi'),
        run_config=RunConfig(telemetry=telemetry_config),
    ):
      pass

  await asyncio.gather(
      _run(
          tenant_on,
          TelemetryConfig(genai_semconv_stability_opt_in='experimental'),
      ),
      _run(
          tenant_off, TelemetryConfig(genai_semconv_stability_opt_in='stable')
      ),
  )

  # Bucket experimental records by agent.
  exp_by_agent: dict[str, int] = {}
  for r in _experimental_logs(log_collector):
    agent = (r.attributes or {}).get('gen_ai.agent.name')
    if agent is None:
      continue
    exp_by_agent[agent] = exp_by_agent.get(agent, 0) + 1

  assert exp_by_agent.get('tenant_on', 0) >= 1, (
      "tenant_on (genai_semconv_stability_opt_in='experimental') did NOT emit"
      f' any {_EXPERIMENTAL_LOG_EVENT!r} LogRecord -- telemetry_config was'
      f' dropped or leaked into tenant_off. exp_by_agent={exp_by_agent}'
  )
  assert exp_by_agent.get('tenant_off', 0) == 0, (
      "tenant_off (genai_semconv_stability_opt_in='stable') unexpectedly"
      f" emitted an {_EXPERIMENTAL_LOG_EVENT!r} LogRecord -- tenant_on's"
      f' telemetry_config bled into tenant_off. exp_by_agent={exp_by_agent}'
  )

  # Stable records don't carry agent identity, so we can only assert
  # globally; tenant_off is the only opted-out invocation in this test.
  assert _stable_logs(log_collector), (
      "tenant_off (genai_semconv_stability_opt_in='stable') should have"
      ' produced stable-semconv per-message LogRecords'
      f' ({sorted(_STABLE_LOG_EVENTS)}) but none were emitted. emitted'
      f' event_names={sorted({r.event_name for r in log_collector})}'
  )


@pytest.mark.asyncio
async def test_runner_invocation_with_admin_lock_ignores_per_request_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_collector: list,
):
  """Admin lock makes the runner ignore RunConfig.telemetry end-to-end.

  With the lock set, ``OTEL_SEMCONV_STABILITY_OPT_IN`` unset, and a
  per-request ``genai_semconv_stability_opt_in='experimental'``, the
  experimental LogRecord must not appear (the lock wins) and the stable
  per-message LogRecords must appear (the env-var fallback takes effect).
  Mirror image of
  ``test_runner_invocation_with_experimental_telemetry_emits_experimental_log``
  with the admin lock added.

  Args:
    monkeypatch: pytest fixture for setting / unsetting env vars.
    span_exporter: wires a real span SDK so ``maybe_log_completion_details``
      does not bail out on ``span=None``.
    log_collector: collects every LogRecord emitted by ``tracing.otel_logger``
      during the test.
  """
  del span_exporter  # Wired up by fixture; not directly asserted on.
  monkeypatch.setenv(_ENV_ADMIN_LOCK, '1')
  monkeypatch.delenv(_ENV_EXPERIMENTAL, raising=False)
  runner = _make_test_runner()
  session = await runner.runner.session_service.create_session(
      app_name=runner.app_name, user_id='test_user'
  )
  async for _ in runner.runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=UserContent('hi'),
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              genai_semconv_stability_opt_in='experimental'
          )
      ),
  ):
    pass

  exp = _experimental_logs(log_collector)
  stable = _stable_logs(log_collector)
  assert not exp, (
      'admin lock was active but the per-request'
      " genai_semconv_stability_opt_in='experimental'"
      f' still produced {len(exp)} {_EXPERIMENTAL_LOG_EVENT!r} LogRecord(s);'
      ' some call site bypassed the lock guard in is_experimental_semconv'
      ' and reached the per-request field directly. Emitted event_names='
      f'{sorted({r.event_name for r in log_collector})}'
  )
  assert stable, (
      'admin lock active + env var unset should fall back to the'
      ' stable-semconv per-message LogRecords'
      f' ({sorted(_STABLE_LOG_EVENTS)}); got none. Emitted event_names='
      f'{sorted({r.event_name for r in log_collector})}'
  )


_GCP_LLM_REQUEST_ATTR = 'gcp.vertex.agent.llm_request'


def _llm_request_span_attrs(
    span_exporter: InMemorySpanExporter,
) -> list[str | int | float | bool | None]:
  """Collects the legacy ADK llm_request span attribute across all spans."""
  return [
      span.attributes.get(_GCP_LLM_REQUEST_ATTR)
      for span in span_exporter.get_finished_spans()
      if span.attributes is not None
      and _GCP_LLM_REQUEST_ATTR in span.attributes
  ]


@pytest.mark.asyncio
async def test_runner_invocation_with_capture_false_elides_legacy_span_attrs(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_collector: list,
):
  """Per-request NO_CONTENT suppresses legacy ADK span attributes.

  With ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=true`` (env default-on) and a
  per-request ``capture_message_content=NO_CONTENT``, the
  ``gcp.vertex.agent.llm_request`` span attribute set by ``trace_call_llm``
  must be elided to ``'{}'``, so the per-request override wins over the env
  var through ``base_llm_flow.py`` -> ``trace_call_llm`` ->
  ``_should_add_request_response_to_spans``.

  Args:
    monkeypatch: pytest fixture for setting / unsetting env vars.
    span_exporter: wires a real span SDK so trace_call_llm emits the attributes
      asserted on here.
    log_collector: collects OTel LogRecords; kept to match the wiring of the
      LogRecord-side tests, not asserted on here.
  """
  del log_collector  # Kept to match otel_logger wiring; not asserted on.
  monkeypatch.setenv(_ENV_ADK_SPAN_CAPTURE, 'true')
  runner = _make_test_runner()
  session = await runner.runner.session_service.create_session(
      app_name=runner.app_name, user_id='test_user'
  )
  async for _ in runner.runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=UserContent('hi'),
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              capture_message_content=ContentCapturingMode.NO_CONTENT
          )
      ),
  ):
    pass

  llm_request_attrs = _llm_request_span_attrs(span_exporter)
  assert llm_request_attrs, (
      f'no spans had the {_GCP_LLM_REQUEST_ATTR!r} attribute set; '
      'trace_call_llm may not have run. spans='
      f'{[s.name for s in span_exporter.get_finished_spans()]}'
  )
  assert all(v == '{}' for v in llm_request_attrs), (
      f"expected all {_GCP_LLM_REQUEST_ATTR!r} attrs to be elided to '{{}}'"
      ' when capture_message_content=ContentCapturingMode.NO_CONTENT;'
      ' per-request override did not reach trace_call_llm. got'
      f' attrs={llm_request_attrs}'
  )


@pytest.mark.asyncio
async def test_runner_invocation_with_admin_lock_ignores_span_capture_override(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_collector: list,
):
  """Admin lock applies end-to-end to legacy ADK span capture too.

  With the lock on, ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false``, and a
  per-request ``capture_message_content=SPAN_AND_EVENT``, the
  ``gcp.vertex.agent.llm_request`` span attribute is elided to ``'{}'``
  (lock + env both say no, override ignored). A bypassed lock guard at the
  ``trace_call_llm`` site would re-enable capture and fail the assertion.

  ``SPAN_AND_EVENT`` is used rather than ``EVENT_ONLY`` because only the
  span-bearing modes (``SPAN_ONLY`` / ``SPAN_AND_EVENT``) flip the legacy
  ADK span knob to True; ``EVENT_ONLY`` would pass trivially even if the
  lock were not honored.

  Args:
    monkeypatch: pytest fixture for setting / unsetting env vars.
    span_exporter: wires a real span SDK.
    log_collector: not asserted on here; see sibling test.
  """
  del log_collector
  monkeypatch.setenv(_ENV_ADMIN_LOCK, '1')
  monkeypatch.setenv(_ENV_ADK_SPAN_CAPTURE, 'false')
  runner = _make_test_runner()
  session = await runner.runner.session_service.create_session(
      app_name=runner.app_name, user_id='test_user'
  )
  async for _ in runner.runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=UserContent('hi'),
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              capture_message_content=ContentCapturingMode.SPAN_AND_EVENT
          )
      ),
  ):
    pass

  llm_request_attrs = _llm_request_span_attrs(span_exporter)
  assert (
      llm_request_attrs
  ), f'no spans had the {_GCP_LLM_REQUEST_ATTR!r} attribute set'
  assert all(v == '{}' for v in llm_request_attrs), (
      'admin lock + env=false should suppress the legacy ADK span'
      ' content attribute regardless of per-request capture=True; some'
      ' call site bypassed the lock guard. attrs={llm_request_attrs}'
  )
