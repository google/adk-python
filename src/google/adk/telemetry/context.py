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

"""Per-request OpenTelemetry configuration types.

:class:`TelemetryConfig` (attached to ``RunConfig.telemetry``) is the single
source of truth for how each telemetry knob resolves. Its ``resolved_*`` /
``should_*`` properties own the precedence ladder (admin lock > per-request
field > ``OTEL_*`` env var > default); the decision functions in
``_experimental_semconv`` and ``tracing`` are thin wrappers over them.

Setting ``ADK_TELEMETRY_IGNORE_RUN_CONFIG`` to ``'1'`` / ``'true'`` makes the
properties ignore the per-request fields and fall back to the env vars.
"""

from __future__ import annotations

import enum
import os
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

ADK_TELEMETRY_IGNORE_RUN_CONFIG = 'ADK_TELEMETRY_IGNORE_RUN_CONFIG'
OTEL_SEMCONV_STABILITY_OPT_IN = 'OTEL_SEMCONV_STABILITY_OPT_IN'
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'
)
# Legacy ADK span-content knob; unlike the OTel env var above, it defaults on.
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS = 'ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS'

# Token in OTEL_SEMCONV_STABILITY_OPT_IN that selects experimental GenAI semconv.
_GENAI_EXPERIMENTAL_OPT_IN = 'gen_ai_latest_experimental'

# Env values (lowercased) treated as "on" / "off" for boolean env vars.
_TRUTHY_ENV_VALUES = frozenset({'1', 'true'})
_FALSY_ENV_VALUES = frozenset({'0', 'false'})


class ContentCapturingMode(enum.Enum):
  """Mirror of ``opentelemetry.util.genai.types.ContentCapturingMode``.

  Defined locally rather than imported because ``opentelemetry-util-genai``
  is an optional, in-development dependency. Values are the canonical states
  for ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``.

  Members:
    NO_CONTENT: No content captured (matches env value ``''``).
    EVENT_ONLY: Content on the emitted LogRecord only.
    SPAN_ONLY: Content on the active span only.
    SPAN_AND_EVENT: Content on both the LogRecord and the active span.
  """

  NO_CONTENT = 'NO_CONTENT'
  EVENT_ONLY = 'EVENT_ONLY'
  SPAN_ONLY = 'SPAN_ONLY'
  SPAN_AND_EVENT = 'SPAN_AND_EVENT'


def _is_span_bearing(mode: ContentCapturingMode) -> bool:
  """Whether ``mode`` routes content onto the span (``SPAN_ONLY`` / ``SPAN_AND_EVENT``)."""
  return mode in (
      ContentCapturingMode.SPAN_ONLY,
      ContentCapturingMode.SPAN_AND_EVENT,
  )


class TelemetryConfig(BaseModel):
  """Per-request OpenTelemetry configuration.

  Attached to an invocation via ``RunConfig.telemetry``. Any field left as
  ``None`` falls back to its corresponding env var (an ``OTEL_*`` var, plus the
  default-on ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`` for legacy spans).
  ``frozen=True`` lets the same config be shared safely across concurrent
  invocations; the resolution properties read env lazily, so later
  ``os.environ`` changes are still picked up.

  Limitations:
    * When ``opentelemetry-instrumentation-google-genai`` is installed and
      wraps ``google.genai.Models.generate_content``, span creation is
      delegated to that library, which reads its own OTel env vars; per-request
      overrides are inoperative for the inference span (but still apply to
      ADK-owned spans).

  Attributes:
    genai_semconv_stability_opt_in: Override for
      ``OTEL_SEMCONV_STABILITY_OPT_IN``. ``'experimental'`` opts in to the
      experimental GenAI semconv attributes; ``'stable'`` keeps the legacy path.
      ``'stable'`` has no env-var equivalent (the env path infers stable from
      the absence of ``'gen_ai_latest_experimental'`` in the CSV).
    capture_message_content: Override for
      ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Pass a
      :class:`ContentCapturingMode` member; the env-var path accepts the
      matching uppercase string.
  """

  model_config = ConfigDict(frozen=True, extra='forbid')

  genai_semconv_stability_opt_in: Optional[
      Literal['stable', 'experimental']
  ] = None
  capture_message_content: Optional[ContentCapturingMode] = None

  @property
  def _ignore_per_request(self) -> bool:
    """Whether the admin lock (``ADK_TELEMETRY_IGNORE_RUN_CONFIG``) is set.

    When set, the per-request fields are ignored and resolution falls back to
    the ``OTEL_*`` env vars.
    """
    lock = os.getenv(ADK_TELEMETRY_IGNORE_RUN_CONFIG, '').strip().lower()
    return lock in _TRUTHY_ENV_VALUES

  @property
  def should_use_experimental_genai_semconv(self) -> bool:
    """Whether to emit experimental GenAI semconv attributes.

    Precedence: admin lock > ``genai_semconv_stability_opt_in`` >
    ``OTEL_SEMCONV_STABILITY_OPT_IN`` env var > ``False``.
    """
    if (
        not self._ignore_per_request
        and self.genai_semconv_stability_opt_in is not None
    ):
      return self.genai_semconv_stability_opt_in == 'experimental'
    opt_ins = os.getenv(OTEL_SEMCONV_STABILITY_OPT_IN)
    if not opt_ins:
      return False
    return _GENAI_EXPERIMENTAL_OPT_IN in (x.strip() for x in opt_ins.split(','))

  @property
  def resolved_content_capturing_mode(self) -> ContentCapturingMode:
    """The effective GenAI content-capturing mode.

    Precedence: admin lock > ``capture_message_content`` >
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` env var (legacy
    ``'true'`` / ``'1'`` coerce to ``EVENT_ONLY``) > ``NO_CONTENT``. Env values
    outside the four-state set fall back to ``NO_CONTENT``.
    """
    if (
        not self._ignore_per_request
        and self.capture_message_content is not None
    ):
      return self.capture_message_content
    stripped = os.getenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, ''
    ).strip()
    # Back-compat: the old env path was boolean; a truthy value means EVENT_ONLY.
    if stripped.lower() in _TRUTHY_ENV_VALUES:
      return ContentCapturingMode.EVENT_ONLY
    try:
      return ContentCapturingMode(stripped.upper())
    except ValueError:
      return ContentCapturingMode.NO_CONTENT

  @property
  def content_capturing_mode_value(self) -> str:
    """:attr:`resolved_content_capturing_mode` as the canonical string.

    Returns ``''`` for ``NO_CONTENT`` (matching the historical env-var
    contract) and the member value otherwise.
    """
    mode = self.resolved_content_capturing_mode
    return '' if mode is ContentCapturingMode.NO_CONTENT else mode.value

  @property
  def should_add_content_to_logs(self) -> bool:
    """Whether content goes on emitted LogRecords (``EVENT_ONLY`` / ``SPAN_AND_EVENT``)."""
    return self.resolved_content_capturing_mode in (
        ContentCapturingMode.EVENT_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )

  @property
  def should_add_content_to_experimental_spans(self) -> bool:
    """Whether content goes on the experimental inference span.

    OTel-spec routing: true for the span-bearing modes (``SPAN_ONLY`` /
    ``SPAN_AND_EVENT``). Distinct from the legacy ADK knob in
    :attr:`should_add_content_to_legacy_spans`, which has its own env fallback.
    """
    return _is_span_bearing(self.resolved_content_capturing_mode)

  @property
  def should_add_content_to_legacy_spans(self) -> bool:
    """Whether content goes on ADK-owned (legacy) spans.

    Separate knob from the OTel content env var. A per-request
    ``capture_message_content`` uses the OTel-spec span routing; otherwise this
    falls back to ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS``, which defaults on.
    """
    if (
        not self._ignore_per_request
        and self.capture_message_content is not None
    ):
      return _is_span_bearing(self.capture_message_content)
    env_value = (
        os.getenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'true').strip().lower()
    )
    return env_value not in _FALSY_ENV_VALUES
