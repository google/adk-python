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

"""Tests for the ContainerCodeExecutor container hardening defaults."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from unittest import mock

from google.adk.code_executors import container_code_executor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.container_code_executor import ContainerCodeExecutor
import pydantic
import pytest


def _mock_docker_client():
  """Returns a mock Docker client whose container passes python verification."""
  client = mock.MagicMock()
  container = mock.MagicMock()
  # `_verify_python_installation` runs `exec_run(['which', 'python3'])` and
  # checks `exit_code == 0`.
  container.exec_run.return_value = mock.MagicMock(exit_code=0)
  client.containers.run.return_value = container
  return client


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_container_is_hardened_by_default(mock_docker):
  """Networking is disabled and privileges are dropped by default."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client

  ContainerCodeExecutor(image='test-image')

  _, kwargs = client.containers.run.call_args
  # Untrusted model-generated code must not be able to reach the network
  # (e.g. the cloud metadata endpoint) or escalate privileges by default.
  assert kwargs['network_disabled']
  assert kwargs['cap_drop'] == ['ALL']
  assert kwargs['security_opt'] == ['no-new-privileges']


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_container_network_can_be_explicitly_enabled(mock_docker):
  """Networking is left enabled when the caller opts in."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client

  ContainerCodeExecutor(image='test-image', network_enabled=True)

  _, kwargs = client.containers.run.call_args
  assert not kwargs['network_disabled']


def _executed_command(container) -> list[str]:
  """Returns the command of the last `exec_run` call on the container."""
  args, _ = container.exec_run.call_args
  return args[0]


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_bounds_execution_by_default(mock_docker):
  """Code runs under a finite timeout even when the caller sets none."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image')
  container = client.containers.run.return_value
  container.exec_run.return_value = mock.MagicMock(
      exit_code=0, output=(b'', b'')
  )

  executor.execute_code(mock.MagicMock(), CodeExecutionInput(code='x = 1'))

  # The container is shared by every invocation, so an unbounded run would pin
  # it for all later callers.
  assert executor.timeout_seconds == 300
  assert _executed_command(container) == [
      'python3',
      '-c',
      container_code_executor._TIMEOUT_WRAPPER,
      '300',
      'x = 1',
  ]


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_passes_configured_timeout(mock_docker):
  """The inherited `timeout_seconds` bounds the in-container execution."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image', timeout_seconds=7)
  container = client.containers.run.return_value
  container.exec_run.return_value = mock.MagicMock(
      exit_code=0, output=(b'', b'')
  )

  executor.execute_code(
      mock.MagicMock(), CodeExecutionInput(code='while True: pass')
  )

  assert _executed_command(container) == [
      'python3',
      '-c',
      container_code_executor._TIMEOUT_WRAPPER,
      '7',
      'while True: pass',
  ]


@mock.patch('google.adk.code_executors.container_code_executor.docker')
@pytest.mark.parametrize('timeout', [0, -1, None])
def test_non_positive_timeout_is_rejected(mock_docker, timeout):
  """A timeout of 0 or None would mean no bound, so it is refused up front."""
  mock_docker.from_env.return_value = _mock_docker_client()

  with pytest.raises(pydantic.ValidationError):
    ContainerCodeExecutor(image='test-image', timeout_seconds=timeout)


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_reports_timeout(mock_docker):
  """A run the supervisor cut short is reported as a timeout."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image', timeout_seconds=7)
  container = client.containers.run.return_value
  container.exec_run.return_value = mock.MagicMock(
      exit_code=container_code_executor._TIMEOUT_EXIT_CODE, output=(b'', b'')
  )

  result = executor.execute_code(
      mock.MagicMock(), CodeExecutionInput(code='while True: pass')
  )

  assert 'timed out after 7 seconds' in result.stderr


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_reports_timeout_alongside_stderr(mock_docker):
  """Output written before the alarm fired does not hide the timeout."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image', timeout_seconds=7)
  container = client.containers.run.return_value
  container.exec_run.return_value = mock.MagicMock(
      exit_code=container_code_executor._TIMEOUT_EXIT_CODE,
      output=(b'', b'a warning from the code'),
  )

  result = executor.execute_code(
      mock.MagicMock(), CodeExecutionInput(code='while True: pass')
  )

  assert 'a warning from the code' in result.stderr
  assert 'timed out after 7 seconds' in result.stderr


# The wrapper below is a string of Python that only ever runs inside the
# container, so the tests run it directly on this host instead: no docker, no
# daemon, and only snippets written here.
_POSIX_ONLY = pytest.mark.skipif(
    not hasattr(os, 'fork') or not hasattr(os, 'killpg'),
    reason='The in-container bound is enforced with POSIX process groups.',
)


def _run_wrapper(timeout: int, code: str) -> subprocess.CompletedProcess:
  """Runs the wrapper exactly as the container executor asks the container to."""
  return subprocess.run(
      [
          sys.executable,
          '-c',
          container_code_executor._TIMEOUT_WRAPPER,
          str(timeout),
          code,
      ],
      capture_output=True,
      text=True,
      timeout=30,
      check=False,
  )


def _is_alive(pid: int) -> bool:
  """Returns whether `pid` is a live (non-zombie) process."""
  try:
    with open(f'/proc/{pid}/stat', encoding='utf-8') as stat_file:
      state = stat_file.read().rsplit(')', 1)[1].split()[0]
  except OSError:
    return False
  return state != 'Z'


@_POSIX_ONLY
def test_wrapper_kills_a_run_that_hits_the_bound():
  """A loop that never returns is killed, not merely waited on."""
  started = time.monotonic()

  completed = _run_wrapper(1, 'while True: pass')

  assert completed.returncode == container_code_executor._TIMEOUT_EXIT_CODE
  assert time.monotonic() - started < 15


@_POSIX_ONLY
def test_wrapper_bound_cannot_be_disarmed_by_the_executed_code():
  """The deadline is held by a process the executed code never runs in."""
  completed = _run_wrapper(
      1,
      'import signal, time\n'
      'signal.alarm(0)\n'
      'signal.signal(signal.SIGALRM, signal.SIG_IGN)\n'
      'time.sleep(25)\n'
      'print("outlived the bound")\n',
  )

  assert completed.returncode == container_code_executor._TIMEOUT_EXIT_CODE
  assert 'outlived the bound' not in completed.stdout


@_POSIX_ONLY
@pytest.mark.skipif(
    not os.path.isdir('/proc'), reason='Liveness is checked through /proc.'
)
def test_wrapper_kills_what_the_code_spawned(tmp_path):
  """Whatever the run started dies with it, so the container is not pinned."""
  pid_file = tmp_path / 'spawned.pid'
  code = textwrap.dedent(f"""
      import os
      import time

      spawned = os.fork()
      if spawned == 0:
        time.sleep(60)
        os._exit(0)
      with open({str(pid_file)!r}, 'w') as f:
        f.write(str(spawned))
      time.sleep(60)
      """)

  completed = _run_wrapper(1, code)

  assert completed.returncode == container_code_executor._TIMEOUT_EXIT_CODE
  assert pid_file.exists(), 'the code never got as far as spawning a process'
  spawned_pid = int(pid_file.read_text())
  try:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _is_alive(spawned_pid):
      time.sleep(0.05)
    assert not _is_alive(spawned_pid)
  finally:
    try:
      os.kill(spawned_pid, signal.SIGKILL)
    except OSError:
      pass


@_POSIX_ONLY
def test_wrapper_leaves_argv_as_a_plain_python_c_run_would():
  """The timeout and the source do not leak into the code's own arguments."""
  completed = _run_wrapper(
      5,
      'import argparse, sys\n'
      'argparse.ArgumentParser().parse_args()\n'
      'print(sys.argv)\n',
  )

  assert completed.returncode == 0, completed.stderr
  assert completed.stdout.strip() == "['-c']"


@_POSIX_ONLY
def test_wrapper_passes_through_output_and_exit_code():
  """A run that finishes on its own is reported exactly as it ended."""
  completed = _run_wrapper(5, 'import sys\nprint("hello")\nsys.exit(3)\n')

  assert completed.returncode == 3
  assert completed.stdout.strip() == 'hello'


@_POSIX_ONLY
def test_wrapper_reports_an_uncaught_error_against_the_original_line():
  """Wrapping the code does not shift the line numbers in its traceback."""
  completed = _run_wrapper(5, 'x = 1\nraise ValueError("boom")\n')

  assert completed.returncode == 1
  assert '"<adk_code>", line 2' in completed.stderr
  assert 'boom' in completed.stderr
