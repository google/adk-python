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
import subprocess
import sys

from pydantic import Field
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...code_executors.base_code_executor import BaseCodeExecutor
from ...code_executors.code_execution_utils import CodeExecutionInput
from ...code_executors.code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)


def _filter_stderr(stderr: str | None) -> str:
  """Filters out harmless sandbox warning messages from stderr."""
  if not stderr:
    return ''
  filtered_lines = []
  for line in stderr.splitlines():
    # Filter out the harmless netns cleanup warnings
    if (
        'Failed to cleanup network namespace' in line
        or 'failed to unmount netns file' in line
    ):
      continue
    filtered_lines.append(line)
  return '\n'.join(filtered_lines)


class CloudRunSandboxCodeExecutor(BaseCodeExecutor):
  """Executes Python code inside a Cloud Run sandbox using the `sandbox` CLI tool.

  This executor is designed to run from within a Cloud Run container where
  sandboxes are enabled. It cannot be used to execute code remotely from a
  local machine or other external environments, as it relies on the local guest
  `sandbox` binary provided by the Cloud Run container runtime.

  It executes the code by passing it via stdin to the Python interpreter
  running inside the local sandbox: `sandbox do <python_path>`.
  """

  sandbox_bin: str = '/usr/local/gcp/bin/sandbox'
  """The path to the sandbox binary. Defaults to '/usr/local/gcp/bin/sandbox'."""

  allow_egress: bool = False
  """Whether to allow egress for the sandbox."""

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  def __init__(self, **data):
    if 'stateful' in data and data['stateful']:
      raise ValueError(
          'Cannot set `stateful=True` in CloudRunSandboxCodeExecutor.'
      )
    if 'optimize_data_file' in data and data['optimize_data_file']:
      raise ValueError(
          'Cannot set `optimize_data_file=True` in CloudRunSandboxCodeExecutor.'
      )

    super().__init__(**data)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    logger.debug(
        'Executing code in Cloud Run Sandbox:\n```\n%s\n```',
        code_execution_input.code,
    )

    # Construct the sandbox command
    # We use 'sandbox do' to run the command in a one-shot sandbox.
    # By default, 'sandbox do' mounts the host's rootfs as read-only, which is fine
    # since we are passing the code via stdin and don't need to read host files,
    # but the python3 binary and libraries from the host rootfs are available.
    cmd = [self.sandbox_bin, 'do']
    if self.allow_egress:
      cmd.append('--allow-egress')

    # We run the same python binary as the current process, using its absolute path
    # to avoid PATH resolution issues inside the sandbox (where PATH might be empty).
    cmd.append(sys.executable or 'python3')

    logger.debug('Running sandbox command: %s', ' '.join(cmd))

    timeout = self.timeout_seconds if self.timeout_seconds is not None else None

    try:
      # Run the command and capture output, writing the code to stdin
      result = subprocess.run(
          cmd,
          input=code_execution_input.code,
          capture_output=True,
          text=True,
          timeout=timeout,
          check=False,
      )

      logger.debug(
          'Sandbox execution finished. Return code: %d, Stdout len: %d, Stderr'
          ' len: %d',
          result.returncode,
          len(result.stdout) if result.stdout else 0,
          len(result.stderr) if result.stderr else 0,
      )
      if result.stderr:
        logger.warning('Sandbox stderr: %s', result.stderr)

      stderr_filtered = _filter_stderr(result.stderr)
      return CodeExecutionResult(
          stdout=result.stdout,
          stderr=stderr_filtered,
          output_files=[],
      )

    except subprocess.TimeoutExpired as e:
      logger.error('Sandbox execution timed out: %s', e)
      # TimeoutExpired.output/stderr might be bytes or str depending on how it was run,
      # but since we passed text=True, they should be str if captured.
      # However, they might be None if no output was captured before timeout.
      stdout_str = (
          e.output
          if isinstance(e.output, str)
          else (e.output.decode('utf-8') if e.output else '')
      )
      stderr_str = (
          e.stderr
          if isinstance(e.stderr, str)
          else (e.stderr.decode('utf-8') if e.stderr else '')
      )
      stderr_filtered = _filter_stderr(stderr_str)
      return CodeExecutionResult(
          stdout=stdout_str,
          stderr=stderr_filtered
          or f'Code execution timed out after {self.timeout_seconds} seconds.',
          output_files=[],
      )
    except FileNotFoundError as e:
      logger.error('Sandbox binary not found: %s', e)
      return CodeExecutionResult(
          stdout='',
          stderr=(
              f'Sandbox binary "{self.sandbox_bin}" not found. Ensure you are'
              ' running in an environment with the sandbox tool installed.'
          ),
          output_files=[],
      )
    except Exception as e:
      logger.error('Unexpected error running sandbox: %s', e)
      return CodeExecutionResult(
          stdout='',
          stderr=f'Unexpected error running sandbox: {e}',
          output_files=[],
      )
