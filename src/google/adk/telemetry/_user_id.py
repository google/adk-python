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

"""Propagation of the ADK session ``user.id`` onto GenAI telemetry records.

``user.id`` is propagated on the OTel context for the duration of an inference
span and copied onto the relevant log records by an installed
``LogRecordProcessor``. This single mechanism serves both inference paths:

* the ADK-native path, where ADK emits the records itself, and
* the delegated path, where ``opentelemetry-instrumentation-google-genai``
  owns the span and the records and ignores the keys ADK stashes on the
  context, so ADK cannot tag the records directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import MutableMapping
from contextlib import contextmanager
import logging
import threading
from typing import TYPE_CHECKING

from opentelemetry import context as otel_context
from opentelemetry._logs import get_logger_provider
from opentelemetry.context import Context
from opentelemetry.sdk._logs import LogRecordProcessor
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
from typing_extensions import override

from ._experimental_semconv import COMPLETION_DETAILS_EVENT_NAME
from ._stable_semconv import GEN_AI_USER_MESSAGE_EVENT

if TYPE_CHECKING:
  from opentelemetry.sdk._logs import ReadWriteLogRecord

  from .context import TelemetryConfig

logger = logging.getLogger("google_adk." + __name__)

# Unique, process-stable key under which the user id is stashed on the OTel
# context. ``create_key`` appends a uuid, so the key cannot collide with keys
# created elsewhere.
_USER_ID_CONTEXT_KEY = otel_context.create_key("adk-gen-ai-user-id")

# Event names whose records carry user-authored content and should therefore be
# tagged with ``user.id``. Other records (e.g. ``gen_ai.system.message``,
# ``gen_ai.choice``) are deliberately left untouched so PII is not sprayed
# across every emitted record.
_USER_ID_EVENT_ALLOWLIST = frozenset({
    GEN_AI_USER_MESSAGE_EVENT,
    COMPLETION_DETAILS_EVENT_NAME,
})

# Guards a single global install of the LogRecordProcessor.
_install_lock = threading.Lock()
_processor_installed = False


@contextmanager
def maybe_propagate_user_id_to_records(
    user_id: str | None,
    telemetry_config: TelemetryConfig,
) -> Iterator[None]:
  """Stashes ``user_id`` on the OTel context for the user-id LogRecordProcessor.

  Wraps the whole inference span (both the ADK-native and the delegated paths).
  The installed ``_UserIdLogRecordProcessor`` reads the value back off each log
  record's captured context and copies it onto ``user.id``. A no-op when there
  is no user id or when the per-request config disables content-bearing logs, so
  ``user.id`` is only attached when message content is also being captured.
  """
  _maybe_install_log_record_processor()

  if user_id is None or not telemetry_config.should_add_content_to_logs:
    yield
    return
  token = otel_context.attach(
      otel_context.set_value(_USER_ID_CONTEXT_KEY, user_id)
  )
  try:
    yield
  finally:
    otel_context.detach(token)


def _get_from_context(context: Context | None) -> str | None:
  """Type-safe read of the propagated user id from ``context``."""
  value = otel_context.get_value(_USER_ID_CONTEXT_KEY, context)
  return value if isinstance(value, str) else None


class _UserIdLogRecordProcessor(LogRecordProcessor):
  """Copies the context-propagated ``user.id`` onto allowlisted log records.

  The records are emitted while the user-id context is active, and the OTel
  ``LogRecord`` snapshots that context at construction, so the user id is
  recoverable here from the record's captured context. Records emitted outside
  an active user-id context, or whose event name is not allowlisted, are left
  untouched.
  """

  @override
  def on_emit(self, log_record: ReadWriteLogRecord) -> None:
    record = log_record.log_record
    if record.event_name not in _USER_ID_EVENT_ALLOWLIST:
      return
    user_id = _get_from_context(record.context)
    if user_id is None:
      return

    if isinstance(record.attributes, MutableMapping):
      record.attributes[USER_ID] = user_id
    else:
      record.attributes = {
          **(record.attributes or {}),
          USER_ID: user_id,
      }

  @override
  def shutdown(self) -> None:
    pass

  @override
  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True


def _maybe_install_log_record_processor() -> None:
  """Installs the user-id LogRecordProcessor once for the process.

  Idempotent: a no-op after the first successful install. Also a no-op while the
  global logger provider is still the API-only no-op/proxy provider (which has
  no ``add_log_record_processor``); in that case a later call retries once an
  SDK logger provider is configured.
  """
  global _processor_installed
  if _processor_installed:
    return
  with _install_lock:
    if _processor_installed:
      return
    provider = get_logger_provider()
    add_processor = getattr(provider, "add_log_record_processor", None)
    if add_processor is None:
      return
    add_processor(_UserIdLogRecordProcessor())
    _processor_installed = True
