# Copyright 2025 Google LLC
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

from __future__ import annotations

from importlib import reload
import io
import json
import logging
from unittest import mock
from unittest import TestCase

from google.adk.cli.utils import logs as logs_module

logs = reload(logs_module)


class TestSetupAdkLogger(TestCase):
  """Tests for setup_adk_logger helper."""

  def setUp(self):
    super().setUp()
    self._reset_logging()
    self.addCleanup(self._reset_logging)

  def _reset_logging(self):
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
      root_logger.removeHandler(handler)
      try:
        handler.close()
      except Exception:  # pylint: disable=broad-except
        pass
    root_logger.setLevel(logging.WARNING)
    root_logger.propagate = True

  def test_log_to_cloud_uses_google_cloud_logging_client(self):
    """Log setup delegates to google.cloud.logging client when available."""
    buffer = io.StringIO()

    class _JsonFormatter(logging.Formatter):

      def format(self, record):
        return json.dumps({'message': record.getMessage()}, ensure_ascii=False)

    class FakeClient:

      def __init__(self):
        self.called_with = None

      def setup_logging(self, log_level=logging.INFO):
        self.called_with = log_level
        handler = logging.StreamHandler(buffer)
        handler.setLevel(log_level)
        handler.setFormatter(_JsonFormatter())
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(log_level)
        root.propagate = False

    fake_client = FakeClient()
    client_factory = mock.Mock(return_value=fake_client)
    with mock.patch.object(logs, 'cloud_logging', autospec=True) as mock_module:
      mock_module.Client = client_factory
      logs.setup_adk_logger(level=logging.INFO, log_to_cloud=True)
      logging.getLogger('google_adk.test').info('hello\nworld')

    client_factory.assert_called_once()
    self.assertEqual(logging.getLogger('google_adk').level, logging.INFO)
    self.assertEqual(fake_client.called_with, logging.INFO)
    output_lines = [
        line for line in buffer.getvalue().splitlines() if line.strip()
    ]
    self.assertEqual(len(output_lines), 1)
    entry = json.loads(output_lines[0])
    self.assertEqual(entry['message'], 'hello\nworld')

  def test_log_to_cloud_client_failure_surfaces_error(self):
    """Cloud logging setup failures surface as actionable errors."""

    class FailingClient:

      def setup_logging(self, log_level=logging.INFO):
        del log_level
        raise OSError('boom')

    client_factory = mock.Mock(return_value=FailingClient())
    with mock.patch.object(logs, 'cloud_logging', autospec=True) as mock_module:
      mock_module.Client = client_factory
      with self.assertRaises(OSError):
        logs.setup_adk_logger(level=logging.INFO, log_to_cloud=True)

    client_factory.assert_called_once()

  def test_text_logging_configures_basic_logging(self):
    """Fallback text logging configures default formatter and handlers."""
    logs.setup_adk_logger(level=logging.ERROR, log_to_cloud=False)

    root_logger = logging.getLogger()
    self.assertEqual(root_logger.level, logging.ERROR)
    self.assertTrue(root_logger.handlers)
    handler = root_logger.handlers[0]
    formatter = handler.formatter
    self.assertIsInstance(formatter, logging.Formatter)
    self.assertEqual(
        formatter._style._fmt,  # pylint: disable=protected-access
        logs.LOGGING_FORMAT,
    )

  def test_text_logging_sets_adk_logger_level(self):
    """ADK logger level is aligned when text logging is used."""
    logs.setup_adk_logger(level=logging.WARNING, log_to_cloud=False)
    self.assertEqual(logging.getLogger('google_adk').level, logging.WARNING)
