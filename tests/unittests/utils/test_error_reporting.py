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

"""Tests for google.adk.utils._error_reporting."""

from __future__ import annotations

import logging

from google.adk.utils import _error_reporting
import pytest

_SECRET = "[Errno 2] No such file: '/home/victim/.adk/credentials.json'"


class TestNewErrorId:

  def test_is_short_and_opaque(self):
    error_id = _error_reporting.new_error_id()

    assert len(error_id) == 12
    assert error_id.isalnum()

  def test_is_unique_per_call(self):
    assert _error_reporting.new_error_id() != _error_reporting.new_error_id()


class TestRedactError:

  def test_omits_the_exception_by_default(self):
    detail = _error_reporting.redact_error(
        FileNotFoundError(_SECRET), summary="Boom.", error_id="abc123"
    )

    assert detail == "Boom. (error_id: abc123)"

  @pytest.mark.parametrize("value", ["1", "true", "TRUE"])
  def test_debug_flag_restores_the_exception(self, monkeypatch, value):
    monkeypatch.setenv(_error_reporting.ENV_DEBUG_ERRORS, value)

    detail = _error_reporting.redact_error(
        FileNotFoundError(_SECRET), summary="Boom.", error_id="abc123"
    )

    assert detail == f"Boom. (error_id: abc123): FileNotFoundError: {_SECRET}"

  @pytest.mark.parametrize("value", ["", "0", "false", "no"])
  def test_non_truthy_debug_flag_keeps_redaction(self, monkeypatch, value):
    monkeypatch.setenv(_error_reporting.ENV_DEBUG_ERRORS, value)

    detail = _error_reporting.redact_error(
        FileNotFoundError(_SECRET), summary="Boom.", error_id="abc123"
    )

    assert detail == "Boom. (error_id: abc123)"


class TestReportError:

  def test_returns_redacted_detail(self, caplog):
    logger = logging.getLogger("google_adk.test_error_reporting")

    with caplog.at_level(logging.ERROR, logger=logger.name):
      detail = _error_reporting.report_error(
          logger, FileNotFoundError(_SECRET), summary="Boom."
      )

    assert detail.startswith("Boom. (error_id: ")
    assert _SECRET not in detail

  def test_logs_the_exception_under_the_returned_id(self, caplog):
    logger = logging.getLogger("google_adk.test_error_reporting")
    exc = FileNotFoundError(_SECRET)

    with caplog.at_level(logging.ERROR, logger=logger.name):
      detail = _error_reporting.report_error(
          logger, exc, summary="Boom.", log_message="Loading the agent failed"
      )

    error_id = detail.split("error_id: ")[1].rstrip(")")
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == (
        f"Loading the agent failed [error_id={error_id}]"
    )
    # The operator handed ``error_id`` must be able to reach the real cause.
    assert record.exc_info[1] is exc

  def test_log_message_defaults_to_the_summary(self, caplog):
    logger = logging.getLogger("google_adk.test_error_reporting")

    with caplog.at_level(logging.ERROR, logger=logger.name):
      _error_reporting.report_error(logger, ValueError("x"), summary="Boom.")

    assert caplog.records[0].getMessage().startswith("Boom. [error_id=")
