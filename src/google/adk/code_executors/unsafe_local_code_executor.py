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

from contextlib import redirect_stdout
import io
import logging
import os
import re
import tempfile
import threading
from typing import Any

from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)

_execution_lock = threading.Lock()


def _prepare_globals(code: str, globals_: dict[str, Any]) -> None:
  """Prepare globals for code execution, injecting __name__ if needed."""
  if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
    globals_['__name__'] = '__main__'


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context."""

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  def __init__(self, **data):
    """Initializes the UnsafeLocalCodeExecutor."""
    if 'stateful' in data and data['stateful']:
      raise ValueError('Cannot set `stateful=True` in UnsafeLocalCodeExecutor.')
    if 'optimize_data_file' in data and data['optimize_data_file']:
      raise ValueError(
          'Cannot set `optimize_data_file=True` in UnsafeLocalCodeExecutor.'
      )
    super().__init__(**data)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    logger.debug('Executing code:\n```\n%s\n```', code_execution_input.code)
    # Execute the code.
    output = ''
    error = ''

    needs_sandbox = (
        code_execution_input.input_files
        or code_execution_input.working_dir
    )

    if needs_sandbox:
      with _execution_lock:
        original_cwd = os.getcwd()
        try:
          with tempfile.TemporaryDirectory() as temp_dir:
            # Write input files to the temp directory
            for f in code_execution_input.input_files:
              file_path = os.path.join(temp_dir, f.path or f.name)
              os.makedirs(os.path.dirname(file_path), exist_ok=True)
              mode = 'wb' if isinstance(f.content, bytes) else 'w'
              with open(file_path, mode) as out_f:
                out_f.write(f.content)

            # Change working directory
            if code_execution_input.working_dir:
              exec_dir = os.path.join(
                  temp_dir, code_execution_input.working_dir
              )
              os.makedirs(exec_dir, exist_ok=True)
              os.chdir(exec_dir)
            else:
              os.chdir(temp_dir)

            globals_ = {}
            _prepare_globals(code_execution_input.code, globals_)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
              exec(code_execution_input.code, globals_, globals_)
            output = stdout.getvalue()

        except Exception as e:
          error = str(e)
        finally:
          os.chdir(original_cwd)
    else:
      # Original path: no temp dir, no chdir, no lock needed
      try:
        globals_ = {}
        _prepare_globals(code_execution_input.code, globals_)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
          exec(code_execution_input.code, globals_, globals_)
        output = stdout.getvalue()
      except Exception as e:
        error = str(e)

    # Collect the final result.
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )
