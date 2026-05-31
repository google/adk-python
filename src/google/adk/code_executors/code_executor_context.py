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

from __future__ import annotations

"""The persistent context used to configure the code executor."""

import copy
import dataclasses
import datetime
from typing import Any
from typing import Optional

from ..sessions.state import State
from .code_execution_utils import File

_CONTEXT_KEY = '_code_execution_context'
_SESSION_ID_KEY = 'execution_session_id'
_PROCESSED_FILE_NAMES_KEY = 'processed_input_files'
_INPUT_FILE_KEY = '_code_executor_input_files'
_ERROR_COUNT_KEY = '_code_executor_error_counts'

_CODE_EXECUTION_RESULTS_KEY = '_code_execution_results'


class CodeExecutorContext:
  """The persistent context used to configure the code executor."""

  _context: dict[str, Any]

  def __init__(self, session_state: State):
    """Initializes the code executor context.

    Args:
      session_state: The session state to get the code executor context from.
    """
    self._context = self._get_code_executor_context(session_state)
    self._session_state = session_state

  def get_state_delta(self) -> dict[str, Any]:
    """Gets the state delta to update in the persistent session state.

    Returns:
      The state delta to update in the persistent session state.
    """
    context_to_update = copy.deepcopy(self._context)
    return {_CONTEXT_KEY: context_to_update}

  def get_execution_id(self) -> Optional[str]:
    """Gets the session ID for the code executor.

    Returns:
      The session ID for the code executor context.
    """
    if _SESSION_ID_KEY not in self._context:
      return None
    return self._context[_SESSION_ID_KEY]

  def set_execution_id(self, session_id: str):
    """Sets the session ID for the code executor.

    Args:
      session_id: The session ID for the code executor.
    """
    self._context[_SESSION_ID_KEY] = session_id

  def get_processed_file_names(self) -> list[str]:
    """Gets the processed file names from the session state.

    Returns:
      A list of processed file names in the code executor context.
    """
    if _PROCESSED_FILE_NAMES_KEY not in self._context:
      return []
    return self._context[_PROCESSED_FILE_NAMES_KEY]

  def add_processed_file_names(self, file_names: [str]):
    """Adds the processed file name to the session state.

    Args:
      file_names: The processed file names to add to the session state.
    """
    if _PROCESSED_FILE_NAMES_KEY not in self._context:
      self._context[_PROCESSED_FILE_NAMES_KEY] = []
    self._context[_PROCESSED_FILE_NAMES_KEY].extend(file_names)

  def get_input_files(self) -> list[File]:
    """Gets the code executor input file names from the session state.

    Returns:
      A list of input files in the code executor context.
    """
    if _INPUT_FILE_KEY not in self._session_state:
      return []
    return [File(**file) for file in self._session_state[_INPUT_FILE_KEY]]

  def add_input_files(
      self,
      input_files: list[File],
  ):
    """Adds the input files to the code executor context.

    Args:
      input_files: The input files to add to the code executor context.
    """
    if _INPUT_FILE_KEY not in self._session_state:
      self._session_state[_INPUT_FILE_KEY] = []
    for input_file in input_files:
      self._session_state[_INPUT_FILE_KEY].append(
          dataclasses.asdict(input_file)
      )

  def clear_input_files(self):
    """Removes the input files and processed file names to the code executor context."""
    if _INPUT_FILE_KEY in self._session_state:
      self._session_state[_INPUT_FILE_KEY] = []
    if _PROCESSED_FILE_NAMES_KEY in self._context:
      self._context[_PROCESSED_FILE_NAMES_KEY] = []

  def get_error_count(self, invocation_id: str) -> int:
    """Gets the error count from the session state.

    Args:
      invocation_id: The invocation ID to get the error count for.

    Returns:
      The error count for the given invocation ID.
    """
    if _ERROR_COUNT_KEY not in self._session_state:
      return 0
    return self._session_state[_ERROR_COUNT_KEY].get(invocation_id, 0)

  def increment_error_count(self, invocation_id: str):
    """Increments the error count from the session state.

    Args:
      invocation_id: The invocation ID to increment the error count for.
    """
    if _ERROR_COUNT_KEY not in self._session_state:
      self._session_state[_ERROR_COUNT_KEY] = {}
    self._session_state[_ERROR_COUNT_KEY][invocation_id] = (
        self.get_error_count(invocation_id) + 1
    )

  def reset_error_count(self, invocation_id: str):
    """Resets the error count from the session state.

    Args:
      invocation_id: The invocation ID to reset the error count for.
    """
    if _ERROR_COUNT_KEY not in self._session_state:
      return
    if invocation_id in self._session_state[_ERROR_COUNT_KEY]:
      del self._session_state[_ERROR_COUNT_KEY][invocation_id]

  def update_code_execution_result(
      self,
      invocation_id: str,
      code: str,
      result_stdout: str,
      result_stderr: str,
  ):
    """Updates the code execution result.

    Args:
      invocation_id: The invocation ID to update the code execution result for.
      code: The code to execute.
      result_stdout: The standard output of the code execution.
      result_stderr: The standard error of the code execution.
    """
    if _CODE_EXECUTION_RESULTS_KEY not in self._session_state:
      self._session_state[_CODE_EXECUTION_RESULTS_KEY] = {}
    if invocation_id not in self._session_state[_CODE_EXECUTION_RESULTS_KEY]:
      self._session_state[_CODE_EXECUTION_RESULTS_KEY][invocation_id] = []
    self._session_state[_CODE_EXECUTION_RESULTS_KEY][invocation_id].append({
        'code': code,
        'result_stdout': result_stdout,
        'result_stderr': result_stderr,
        'timestamp': int(datetime.datetime.now().timestamp()),
    })

  def _get_code_executor_context(self, session_state: State) -> dict[str, Any]:
    """Gets the code executor context from the session state.

    Args:
      session_state: The session state to get the code executor context from.

    Returns:
      A dict of code executor context.
    """
    if _CONTEXT_KEY not in session_state:
      session_state[_CONTEXT_KEY] = {}
    return session_state[_CONTEXT_KEY]
