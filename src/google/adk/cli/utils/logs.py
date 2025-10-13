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

import logging
import os
import tempfile
import time

from google.api_core import exceptions as api_core_exceptions
from google.auth import exceptions as auth_exceptions
from google.cloud import logging as cloud_logging

LOGGING_FORMAT = (
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def setup_adk_logger(level=logging.INFO, *, log_to_cloud: bool = False):
  """Set up ADK logger with optional Google Cloud Logging integration."""
  root_logger = logging.getLogger()

  if log_to_cloud:
    # Remove the default StreamHandler to avoid duplicate stdout logs.
    root_logger.handlers = []
    client = cloud_logging.Client()
    client.setup_logging(log_level=level)
    root_logger.setLevel(level)
  else:
    if root_logger.handlers:
      # Uvicorn installs handlers ahead of application code, so basicConfig is a no-op.
      formatter = logging.Formatter(LOGGING_FORMAT)
      for handler in root_logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
    else:
      logging.basicConfig(level=level, format=LOGGING_FORMAT)
    root_logger.setLevel(level)

  logging.getLogger('google_adk').setLevel(level)


def log_to_tmp_folder(
    level=logging.INFO,
    *,
    sub_folder: str = 'agents_log',
    log_file_prefix: str = 'agent',
    log_file_timestamp: str = time.strftime('%Y%m%d_%H%M%S'),
):
  """Logs to system temp folder, instead of logging to stderr.

  Args
    sub_folder: str = 'agents_log',
    log_file_prefix: str = 'agent',
    log_file_timestamp: str = time.strftime('%Y%m%d_%H%M%S'),

  Returns
    the log file path.
  """
  log_dir = os.path.join(tempfile.gettempdir(), sub_folder)
  log_filename = f'{log_file_prefix}.{log_file_timestamp}.log'
  log_filepath = os.path.join(log_dir, log_filename)

  os.makedirs(log_dir, exist_ok=True)

  file_handler = logging.FileHandler(log_filepath, mode='w')
  file_handler.setLevel(level)
  file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

  root_logger = logging.getLogger()
  root_logger.setLevel(level)
  root_logger.handlers = []  # Clear handles to disable logging to stderr
  root_logger.addHandler(file_handler)

  print(f'Log setup complete: {log_filepath}')

  latest_log_link = os.path.join(log_dir, f'{log_file_prefix}.latest.log')
  if os.path.islink(latest_log_link):
    os.unlink(latest_log_link)
  os.symlink(log_filepath, latest_log_link)

  print(f'To access latest log: tail -F {latest_log_link}')
  return log_filepath
