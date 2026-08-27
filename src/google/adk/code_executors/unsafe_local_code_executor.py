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

import logging
import os
import re
import signal
import subprocess
import sys
from typing import Any

from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)

# How long to wait for a timed-out execution to exit after SIGTERM before
# escalating to SIGKILL, so that the timeout itself cannot block forever.
_TERMINATE_GRACE_SECONDS = 5

# Runs one program in the child interpreter.
#
# The program arrives on stdin rather than in argv because a single argument is
# capped (at 128 KiB on Linux) and generated programs can carry their own data.
# Reading it leaves stdin at end-of-file, which is what the program would have
# seen before.
#
# The traceback is printed here, without the frame this wrapper contributes, so
# that a failure shows the model its own code and not a file inside this
# package -- a frame it can do nothing about, which then stays in the
# conversation for every later request.
_RUNNER = """
import sys, traceback

_run_name = sys.argv[1]
del sys.argv[1:]

_globals = {'__name__': _run_name} if _run_name else {}
_source = sys.stdin.buffer.read().decode('utf-8')

try:
  exec(compile(_source, '<code>', 'exec'), _globals, _globals)
except SystemExit:
  # The program chose its own exit status, so let it stand rather than
  # reporting a deliberate clean exit as a failure.
  raise
except BaseException as exc:
  _tb = exc.__traceback__
  traceback.print_exception(
      type(exc), exc, _tb.tb_next if _tb else None, file=sys.stderr
  )
  sys.exit(1)
"""


def _run_name(code: str) -> str:
  """Returns the `__name__` the code should run under, or '' for none."""
  if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
    return '__main__'
  return ''


def _signal_group(group: int, sig: int) -> None:
  """Signals every process left in a group, tolerating an empty one."""
  if not hasattr(os, 'killpg'):
    return
  try:
    os.killpg(group, sig)
  except OSError:
    logger.debug('Could not signal the execution process group.')


def _kill_execution(process: subprocess.Popen[str]) -> tuple[str, str]:
  """Kills a timed-out execution, returning what it wrote before it died."""
  # The execution leads its own process group, so its pid is the group that
  # holds whatever it spawned. `terminate` and `kill` below reach the execution
  # itself on the platforms that have no group to signal.
  group = process.pid

  # SIGTERM first, so the code and its children get the same grace period the
  # execution process itself gets before anything is killed outright.
  _signal_group(group, signal.SIGTERM)
  process.terminate()
  try:
    process.wait(_TERMINATE_GRACE_SECONDS)
  except subprocess.TimeoutExpired:
    pass

  # Escalate unconditionally: the execution process exiting says nothing about
  # a child of it that is ignoring SIGTERM, and such a child holds the output
  # pipes open besides.
  _signal_group(group, signal.SIGKILL)
  process.kill()
  try:
    return process.communicate(timeout=_TERMINATE_GRACE_SECONDS)
  except subprocess.TimeoutExpired:
    return '', ''


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context."""

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  def __init__(self, **data: Any) -> None:
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

    env = dict(os.environ)
    # A plain child interpreter starts with its own import path, so pass the
    # agent's along: code importing a module from the application resolved
    # before this and still does.
    env['PYTHONPATH'] = os.pathsep.join(p for p in sys.path if p)
    # Pin the child's output encoding. Its stdout is a pipe, so it would
    # otherwise encode with the host locale and a program printing non-ASCII
    # under a non-UTF-8 locale would die encoding its own output.
    env['PYTHONIOENCODING'] = 'utf-8'

    process = subprocess.Popen(
        [sys.executable, '-c', _RUNNER, _run_name(code_execution_input.code)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        errors='replace',
        env=env,
        # Its own session, so a timeout can take down everything the code
        # started and nothing else. Ignored on Windows, as `os.killpg` is.
        start_new_session=True,
    )

    timed_out = False
    try:
      output, error = process.communicate(
          input=code_execution_input.code, timeout=self.timeout_seconds
      )
    except subprocess.TimeoutExpired:
      output, error = _kill_execution(process)
      timed_out = True

    if timed_out:
      # Appended rather than assigned: whatever the code wrote before it was
      # killed is still the useful diagnostic, but on its own it would hide
      # the fact that the run was cut short.
      note = f'Code execution timed out after {self.timeout_seconds} seconds.'
      error = f'{error}\n{note}' if error else note
    elif process.returncode == 0:
      # A non-empty stderr is what marks the result failed and drives the retry
      # counter, so it has to mean the program failed rather than that the
      # program wrote a warning. The exit status is what says which happened.
      error = ''
    elif not error:
      # The code died without saying why: a signal, or a call to `os._exit`.
      # Reporting nothing would show the model a clean run.
      error = f'Code execution exited with status {process.returncode}.'

    # Collect the final result.
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
        exit_code=process.returncode,
    )
