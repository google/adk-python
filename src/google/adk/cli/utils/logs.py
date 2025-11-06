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
import warnings

import click

LOGGING_FORMAT = (
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def setup_adk_logger(level=logging.INFO):
  # Configure the root logger format and level.
  logging.basicConfig(level=level, format=LOGGING_FORMAT)

  adk_logger = logging.getLogger('google_adk')
  adk_logger.setLevel(level)


def _create_symlink(symlink_path: str, target_path: str) -> bool:
  """Creates a symlink at symlink_path pointing to target_path.

  Returns:
    True if successful, False otherwise.
  """
  try:
    if os.path.islink(symlink_path):
      os.unlink(symlink_path)
    elif os.path.exists(symlink_path):
      warnings.warn(
          'Cannot create symlink for latest log file: file exists at'
          f' {symlink_path}'
      )
      return False
    os.symlink(target_path, symlink_path)
    return True
  except OSError:
    return False


def _try_create_latest_log_symlink(
    log_dir: str, log_file_prefix: str, log_filepath: str
) -> None:
  """Attempts to create a 'latest' symlink and prints access instructions."""
  latest_log_link = os.path.join(log_dir, f'{log_file_prefix}.latest.log')
  if _create_symlink(latest_log_link, log_filepath):
    click.echo(f'To access latest log: tail -F {latest_log_link}')
  else:
    click.echo(f'To access latest log: tail -F {log_filepath}')


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

  click.echo(f'Log setup complete: {log_filepath}')
  _try_create_latest_log_symlink(log_dir, log_file_prefix, log_filepath)

  return log_filepath
