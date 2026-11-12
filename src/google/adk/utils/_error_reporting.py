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

"""Reporting errors to untrusted callers without leaking their detail.

This module is for ADK internal use only.
Please do not rely on the implementation details.

An exception message is written for whoever reads the log, not for whoever sent
the request. ``FileNotFoundError``, ``ImportError``,
``pydantic.ValidationError``
and SDK client errors routinely carry absolute filesystem paths, the layout of
the installed packages, environment variable names, configuration values and
echoed request payloads. Handing that back over the wire tells a caller how the
host is laid out, which is exactly the reconnaissance a path-traversal or
enumeration attempt needs, and the caller cannot act on any of it anyway.

So the exception stays server-side: the full traceback goes to the log next to a
short opaque id, and the caller gets a fixed summary plus that same id. An
operator handling a report greps for the id and lands on the real exception.
"""

from __future__ import annotations

import logging
from typing import Optional
import uuid

from .env_utils import is_env_enabled

ENV_DEBUG_ERRORS = 'ADK_DEBUG_ERRORS'
"""Env var that puts exception text back into responses that leave the process.

Off by default, and meant for local debugging only: enabling it on a
network-reachable deployment restores the disclosure this module exists to
prevent.
"""

_ERROR_ID_LENGTH = 12


def new_error_id() -> str:
  """Returns a short opaque id correlating a response with a log entry."""
  return uuid.uuid4().hex[:_ERROR_ID_LENGTH]


def redact_error(exc: BaseException, *, summary: str, error_id: str) -> str:
  """Builds the error text that is safe to hand to an untrusted caller.

  Args:
    exc: The exception being reported.
    summary: Caller-facing description of what failed. Must be a constant --
      anything interpolated into it is not covered by this redaction.
    error_id: Correlation id that the full exception was logged under.

  Returns:
    ``summary`` and the correlation id, plus -- only when ``ADK_DEBUG_ERRORS``
    is enabled -- the exception type and message.
  """
  detail = f'{summary} (error_id: {error_id})'
  if is_env_enabled(ENV_DEBUG_ERRORS):
    return f'{detail}: {type(exc).__name__}: {exc}'
  return detail


def report_error(
    logger: logging.Logger,
    exc: BaseException,
    *,
    summary: str,
    log_message: Optional[str] = None,
) -> str:
  """Logs an exception in full and returns text safe to return to the caller.

  Args:
    logger: Logger that receives the exception and its traceback.
    exc: The exception being reported.
    summary: Caller-facing description of what failed, e.g. ``'Agent processing
      failed.'``. Must be a constant.
    log_message: Server-side description, defaults to ``summary``.

  Returns:
    The value of :func:`redact_error` for a freshly minted correlation id.
  """
  error_id = new_error_id()
  logger.error(
      '%s [error_id=%s]',
      log_message or summary,
      error_id,
      exc_info=exc,
  )
  return redact_error(exc, summary=summary, error_id=error_id)
