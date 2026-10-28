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

"""Tests for the ContainerCodeExecutor container hardening, timeout, and lifecycle."""

import base64
import gc
import io
import os
import signal
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
from unittest import mock
import weakref

import docker
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors import container_code_executor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import File
from google.adk.code_executors.container_code_executor import ContainerCodeExecutor
from google.adk.platform.thread import create_thread
import pydantic
import pytest


def _mock_docker_client():
  """Returns a mock Docker client whose container passes python verification."""
  client = mock.MagicMock()
  container = mock.MagicMock()
  # `_verify_python_installation` runs `exec_run(['which', 'python3'])` and
  # checks `exit_code == 0`.
  container.exec_run.return_value = mock.MagicMock(
      exit_code=0, output=(b'', b'')
  )
  container.get_archive.return_value = ([b''], mock.MagicMock())
  client.containers.run.return_value = container
  return client


def _set_python3_result(container, **result):
  """Makes the python3 exec return `result`; other execs succeed."""

  def exec_run(cmd, demux=False):
    if cmd[0] == 'python3':
      return mock.MagicMock(**result)
    return mock.MagicMock(exit_code=0, output=(b'', b''))

  container.exec_run.side_effect = exec_run


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
  _set_python3_result(
      container,
      exit_code=container_code_executor._TIMEOUT_EXIT_CODE,
      output=(b'', b''),
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
  _set_python3_result(
      container,
      exit_code=container_code_executor._TIMEOUT_EXIT_CODE,
      output=(b'', b'a warning from the code'),
  )

  result = executor.execute_code(
      mock.MagicMock(), CodeExecutionInput(code='while True: pass')
  )

  assert 'a warning from the code' in result.stderr
  assert 'timed out after 7 seconds' in result.stderr


@mock.patch('google.adk.code_executors.container_code_executor.docker')
@pytest.mark.parametrize('exit_code', [0, 1, 42])
def test_execute_code_reports_the_exit_code(mock_docker, exit_code):
  """The status the container exec returned is passed through verbatim."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image')
  container = client.containers.run.return_value
  _set_python3_result(container, exit_code=exit_code, output=(b'', b''))

  result = executor.execute_code(
      mock.MagicMock(), CodeExecutionInput(code='print(1)')
  )

  assert result.exit_code == exit_code


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


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_explicit_close_stops_and_removes_container(mock_docker):
  """Calling close() explicitly stops and removes the container."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client

  executor = ContainerCodeExecutor(image='test-image')
  ref = weakref.ref(executor)

  executor.close()

  container.stop.assert_called_once()
  container.remove.assert_called_once()
  assert executor._container is None
  assert executor._finalizer is not None
  assert not executor._finalizer.alive

  del executor
  gc.collect()
  assert ref() is None


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_close_is_idempotent(mock_docker):
  """Calling close() multiple times only cleans up the container once."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client

  executor = ContainerCodeExecutor(image='test-image')
  executor.close()
  executor.close()

  container.stop.assert_called_once()
  container.remove.assert_called_once()


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_close_failure_retains_container_and_finalizer(mock_docker):
  """When container stop raises, close retains _container and keeps finalizer alive."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  container.stop.side_effect = [RuntimeError, None]
  mock_docker.from_env.return_value = client

  executor = ContainerCodeExecutor(image='test-image')
  executor.close()

  container.stop.assert_called_once()
  container.remove.assert_not_called()
  assert executor._container is container
  assert executor._finalizer is not None
  assert executor._finalizer.alive

  # Subsequent GC retries cleanup via the live finalizer
  del executor
  gc.collect()

  assert container.stop.call_count == 2
  container.remove.assert_called_once()


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_context_manager_stops_and_removes_container(mock_docker):
  """Using the executor as a context manager cleans up the container on exit."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client

  with ContainerCodeExecutor(image='test-image') as executor:
    assert executor is not None

  container.stop.assert_called_once()
  container.remove.assert_called_once()


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_discarded_executor_is_garbage_collected_and_cleaned_up(mock_docker):
  """Discarded executor without explicit close is garbage collected and cleaned up."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client

  executor = ContainerCodeExecutor(image='test-image')
  ref = weakref.ref(executor)

  del executor
  gc.collect()

  assert ref() is None
  container.stop.assert_called_once()
  container.remove.assert_called_once()


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_init_with_image(mock_docker):
  """Default input and output directories are set for an image executor."""
  mock_docker.from_env.return_value = _mock_docker_client()

  executor = ContainerCodeExecutor(image='test-image:latest')

  assert executor.image == 'test-image:latest'
  assert executor.input_dir == '/tmp/inputs'
  assert executor.output_dir == '/tmp/outputs'


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_init_with_custom_dirs(mock_docker):
  """Custom input and output directories are recorded by Pydantic."""
  mock_docker.from_env.return_value = _mock_docker_client()

  executor = ContainerCodeExecutor(
      image='test-image:latest',
      input_dir='/custom/inputs',
      output_dir='/custom/outputs',
  )

  assert executor.input_dir == '/custom/inputs'
  assert executor.output_dir == '/custom/outputs'
  assert 'input_dir' in executor.model_fields_set
  assert 'output_dir' in executor.model_fields_set


def test_init_requires_image_or_docker_path():
  """Initializing without image or docker_path raises ValueError."""
  with pytest.raises(
      ValueError, match='Either image or docker_path must be set'
  ):
    ContainerCodeExecutor()


def test_init_rejects_stateful():
  """Initializing with stateful=True raises ValueError."""
  with pytest.raises(ValueError, match='Cannot set `stateful=True`'):
    ContainerCodeExecutor(image='test', stateful=True)


def test_init_rejects_optimize_data_file():
  """Initializing with optimize_data_file=True raises ValueError."""
  with pytest.raises(ValueError, match='Cannot set `optimize_data_file=True`'):
    ContainerCodeExecutor(image='test', optimize_data_file=True)


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_basic(mock_docker):
  """Missing output_dir yields stdout and no output files."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  mock_docker.errors = docker.errors
  executor = ContainerCodeExecutor(image='test-image:latest')
  container.exec_run.return_value = mock.MagicMock(
      exit_code=0, output=(b'Hello World', b'')
  )
  container.get_archive.side_effect = docker.errors.APIError(
      'Not found', response=mock.MagicMock(status_code=404)
  )
  context = mock.MagicMock(spec=InvocationContext)

  result = executor.execute_code(
      context, CodeExecutionInput(code='print("Hello World")')
  )

  assert result.stdout == 'Hello World'
  assert result.stderr == ''
  assert result.output_files == []


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_with_error(mock_docker):
  """Failing code returns the stderr of the python3 invocation."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  mock_docker.errors = docker.errors
  executor = ContainerCodeExecutor(image='test-image:latest')

  def exec_run_side_effect(cmd, demux=False):
    if cmd[0] == 'python3':
      return mock.MagicMock(exit_code=1, output=(b'', b'Some error'))
    return mock.MagicMock(exit_code=0, output=(b'', b''))

  container.exec_run.side_effect = exec_run_side_effect
  container.get_archive.side_effect = docker.errors.APIError(
      'Not found', response=mock.MagicMock(status_code=404)
  )
  context = mock.MagicMock(spec=InvocationContext)

  result = executor.execute_code(
      context, CodeExecutionInput(code='raise Error("test")')
  )

  assert result.stderr == 'Some error'
  assert result.exit_code == 1


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_with_input_files(mock_docker):
  """Base64 string and raw bytes input files are archived into input_dir."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  mock_docker.errors = docker.errors
  executor = ContainerCodeExecutor(image='test-image:latest')
  context = mock.MagicMock(spec=InvocationContext)
  binary_bytes = b'\x89PNG\r\n\x1a\n\x00\xff'
  code_input = CodeExecutionInput(
      code='print("test")',
      input_files=[
          File(
              name='image.png',
              content=base64.b64encode(binary_bytes).decode('ascii'),
              mime_type='image/png',
          ),
          File(
              name='notes.txt',
              content=base64.b64encode(b'plain text content').decode(),
          ),
          File(name='raw.bin', content=b'raw byte content'),
      ],
  )

  executor.execute_code(context, code_input)

  container.put_archive.assert_called_once()
  target_dir, archive_bytes = container.put_archive.call_args[0]
  assert target_dir == '/tmp/inputs'
  extracted = {}
  with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r') as tar:
    for member in tar.getmembers():
      file_obj = tar.extractfile(member)
      assert file_obj is not None
      extracted[member.name] = file_obj.read()
  assert extracted == {
      'image.png': binary_bytes,
      'notes.txt': b'plain text content',
      'raw.bin': b'raw byte content',
  }


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_strips_input_file_name_directories(mock_docker):
  """Input file names with path components are reduced to their basename."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')
  context = mock.MagicMock(spec=InvocationContext)

  executor.execute_code(
      context,
      CodeExecutionInput(
          code='print("ok")',
          input_files=[File(name='../../etc/passwd', content=b'content')],
      ),
  )

  _, archive_bytes = container.put_archive.call_args[0]
  with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r') as tar:
    assert tar.getnames() == ['passwd']


@pytest.mark.parametrize(
    'input_files, error',
    [
        ([File(name='..', content=b'content')], 'Invalid input file name'),
        (
            [
                File(name='d1/data.csv', content=b'1'),
                File(name='d2/data.csv', content=b'2'),
            ],
            'Duplicate input file name',
        ),
        ([File(name='notes.txt', content='plain text content')], 'base64'),
    ],
)
@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_rejects_invalid_input_files(
    mock_docker, input_files, error
):
  """Traversal names, duplicate basenames and non-base64 strings raise."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')
  context = mock.MagicMock(spec=InvocationContext)

  with pytest.raises(ValueError, match=error):
    executor.execute_code(
        context, CodeExecutionInput(code='print(1)', input_files=input_files)
    )

  container.put_archive.assert_not_called()


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_with_output_files(mock_docker):
  """Output files keep their path under output_dir and a guessed MIME type."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')

  files_to_archive = [
      ('outputs/output.txt', b'output content'),
      ('outputs/data.csv', b'col1,col2\n1,2\n'),
      ('outputs/unknown.nonexistent', b'\x00\x01\x02'),
      ('outputs/nested/data.csv', b'col3\n3\n'),
  ]
  tar_buffer = io.BytesIO()
  with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
    for name, data in files_to_archive:
      tar_info = tarfile.TarInfo(name=name)
      tar_info.size = len(data)
      tar.addfile(tar_info, io.BytesIO(data))
  tar_bytes = tar_buffer.getvalue()
  container.get_archive.return_value = ([tar_bytes], mock.MagicMock())
  context = mock.MagicMock(spec=InvocationContext)

  result = executor.execute_code(
      context, CodeExecutionInput(code='print("test")')
  )

  assert result.output_files == [
      File(
          name='output.txt',
          content=b'output content',
          mime_type='text/plain',
      ),
      File(
          name='data.csv',
          content=b'col1,col2\n1,2\n',
          mime_type='text/csv',
      ),
      File(
          name='unknown.nonexistent',
          content=b'\x00\x01\x02',
          mime_type='application/octet-stream',
      ),
      File(
          name='nested/data.csv',
          content=b'col3\n3\n',
          mime_type='text/csv',
      ),
  ]


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_reraises_non_404_archive_error(mock_docker):
  """Non-404 APIErrors from get_archive, even without a response, propagate."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  mock_docker.errors = docker.errors
  executor = ContainerCodeExecutor(image='test-image:latest')
  container.get_archive.side_effect = docker.errors.APIError(
      'Daemon error', response=None
  )
  context = mock.MagicMock(spec=InvocationContext)

  with pytest.raises(docker.errors.APIError):
    executor.execute_code(context, CodeExecutionInput(code='print("test")'))


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_clears_directories_between_runs(mock_docker):
  """Input and output files do not persist across executions."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')
  context = mock.MagicMock(spec=InvocationContext)

  container_fs: dict[str, dict[str, bytes]] = {
      '/tmp/inputs': {},
      '/tmp/outputs': {},
  }

  def exec_run_side_effect(cmd, demux=False):
    if cmd[0] == 'rm':
      for path in cmd[2:]:
        container_fs[path] = {}
    elif cmd[0] == 'python3' and 'write_first_output' in cmd[-1]:
      container_fs['/tmp/outputs']['first.txt'] = b'first output'
    return mock.MagicMock(exit_code=0, output=(b'', b''))

  def put_archive_side_effect(path, archive_bytes):
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r') as tar:
      for member in tar.getmembers():
        file_obj = tar.extractfile(member)
        if file_obj is not None:
          container_fs[path][member.name] = file_obj.read()

  def get_archive_side_effect(path):
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
      for name, data in container_fs.get(path, {}).items():
        tar_info = tarfile.TarInfo(name=f'{os.path.basename(path)}/{name}')
        tar_info.size = len(data)
        tar.addfile(tar_info, io.BytesIO(data))
    return ([tar_buffer.getvalue()], mock.MagicMock())

  container.exec_run.side_effect = exec_run_side_effect
  container.put_archive.side_effect = put_archive_side_effect
  container.get_archive.side_effect = get_archive_side_effect

  first_result = executor.execute_code(
      context,
      CodeExecutionInput(
          code='write_first_output',
          input_files=[File(name='input1.txt', content=b'first input')],
      ),
  )
  assert container_fs['/tmp/inputs'] == {'input1.txt': b'first input'}
  assert [f.name for f in first_result.output_files] == ['first.txt']

  second_result = executor.execute_code(
      context,
      CodeExecutionInput(code='print("no files")'),
  )
  assert container_fs['/tmp/inputs'] == {}
  assert second_result.output_files == []


@pytest.mark.parametrize(
    'input_dir, output_dir',
    [
        ('/tmp/io', '/tmp/io/'),
        ('/tmp/io', '/tmp/io/out'),
        ('/tmp/io/in', '/tmp/io'),
    ],
)
@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_init_rejects_overlapping_dirs(mock_docker, input_dir, output_dir):
  """input_dir and output_dir may be neither the same nor nested."""
  mock_docker.from_env.return_value = _mock_docker_client()

  with pytest.raises(ValueError, match='must not be the same or nested'):
    ContainerCodeExecutor(
        image='test-image', input_dir=input_dir, output_dir=output_dir
    )


@pytest.mark.parametrize('failing_cmd', ['rm', 'mkdir'])
@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_raises_when_directories_cannot_be_prepared(
    mock_docker, failing_cmd
):
  """Failing to empty or create the directories raises before the code runs."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')
  executed = []

  def exec_run_side_effect(cmd, demux=False):
    executed.append(cmd[0])
    exit_code = 1 if cmd[0] == failing_cmd else 0
    return mock.MagicMock(exit_code=exit_code, output=b'Permission denied')

  container.exec_run.side_effect = exec_run_side_effect
  context = mock.MagicMock(spec=InvocationContext)

  with pytest.raises(RuntimeError, match='Failed to prepare'):
    executor.execute_code(context, CodeExecutionInput(code='print(1)'))

  assert 'python3' not in executed


@mock.patch.object(container_code_executor, '_MAX_OUTPUT_ARCHIVE_BYTES', 10)
@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_stops_reading_oversized_output(mock_docker):
  """Reading output_dir stops with an error once the size budget is exceeded."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')

  def tar_stream():
    yield b'x' * 8
    yield b'x' * 8
    pytest.fail('Read past the output size budget.')

  container.get_archive.return_value = (tar_stream(), mock.MagicMock())
  context = mock.MagicMock(spec=InvocationContext)

  with pytest.raises(RuntimeError, match='exceed 10 bytes'):
    executor.execute_code(context, CodeExecutionInput(code='print(1)'))


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_rejects_user_scoped_output_name(mock_docker):
  """Output names in the artifact service's user namespace are rejected."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')
  tar_buffer = io.BytesIO()
  with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
    tar_info = tarfile.TarInfo(name='outputs/user:profile.json')
    tar_info.size = 2
    tar.addfile(tar_info, io.BytesIO(b'{}'))
  container.get_archive.return_value = (
      [tar_buffer.getvalue()],
      mock.MagicMock(),
  )
  context = mock.MagicMock(spec=InvocationContext)

  with pytest.raises(RuntimeError, match='reserved for user-scoped'):
    executor.execute_code(context, CodeExecutionInput(code='print(1)'))


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_execute_code_serializes_concurrent_executions(mock_docker):
  """A second execution does not touch the directories while one is running."""
  client = _mock_docker_client()
  container = client.containers.run.return_value
  mock_docker.from_env.return_value = client
  executor = ContainerCodeExecutor(image='test-image:latest')
  counter_lock = threading.Lock()
  active = [0]
  overlapped = threading.Event()

  def exec_run_side_effect(cmd, demux=False):
    if cmd[0] == 'rm':
      with counter_lock:
        active[0] += 1
        if active[0] > 1:
          overlapped.set()
    elif cmd[0] == 'python3':
      overlapped.wait(timeout=0.5)
    return mock.MagicMock(exit_code=0, output=(b'', b''))

  def get_archive_side_effect(path):
    with counter_lock:
      active[0] -= 1
    return ([b''], mock.MagicMock())

  container.exec_run.side_effect = exec_run_side_effect
  container.get_archive.side_effect = get_archive_side_effect
  context = mock.MagicMock(spec=InvocationContext)
  results = []

  def run():
    results.append(
        executor.execute_code(context, CodeExecutionInput(code='print(1)'))
    )

  threads = [create_thread(target=run) for _ in range(2)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  assert len(results) == 2
  assert not overlapped.is_set()
