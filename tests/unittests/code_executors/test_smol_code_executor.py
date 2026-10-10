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

"""Contract tests for the Smol Machines code executor."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.adk.agents import LlmAgent
from google.adk.code_executors import SmolCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import File
import pytest

from tests.unittests import testing_utils


@pytest.fixture
def vm(monkeypatch):
  machine = MagicMock()
  machine.exec.return_value = SimpleNamespace(
      stdout='42\n',
      stderr='',
      exit_code=0,
      stdout_truncated=False,
      stderr_truncated=False,
  )
  create = MagicMock(return_value=machine)

  def options(**kwargs):
    return SimpleNamespace(**kwargs)

  monkeypatch.setitem(
      sys.modules,
      'smol',
      SimpleNamespace(
          Machine=SimpleNamespace(create=create),
          ConnectOptions=options,
          MachineConfig=options,
          ResourceSpec=options,
          ExecOptions=options,
      ),
  )
  return machine, create


def test_local_code_runs_without_guest_egress_and_deletes_vm(vm):
  machine, create = vm
  executor = SmolCodeExecutor(timeout_seconds=7)

  result = executor.execute_code(None, CodeExecutionInput(code='print(42)'))

  assert (result.stdout, result.exit_code) == ('42\n', 0)
  config, conn = create.call_args.args
  assert config.image == 'python:3.12-slim'
  assert config.resources.network is False
  assert config.persistent is False
  assert conn.target == 'local'
  script_path, source = machine.write_file.call_args.args
  assert script_path.startswith('/workspace/adk-code-')
  assert source == 'print(42)'
  assert machine.exec.call_args.args[0] == ['python3', script_path]
  assert machine.exec.call_args.args[1].timeout == 7.0
  machine.delete.assert_called_once_with()


def test_agent_runner_executes_code_and_returns_microvm_output(vm):
  machine, create = vm
  model = testing_utils.MockModel.create(
      responses=['```python\nprint(42)\n```', 'The answer is 42.']
  )
  agent = LlmAgent(
      name='smol_agent', model=model, code_executor=SmolCodeExecutor()
  )

  events = testing_utils.InMemoryRunner(root_agent=agent).run(
      'Calculate the answer with Python.'
  )

  assert any(
      part.code_execution_result and '42' in str(part.code_execution_result)
      for event in events
      if event.content
      for part in event.content.parts or []
  )
  assert events[-1].content.parts[0].text == 'The answer is 42.'
  create.assert_called_once()
  machine.delete.assert_called_once_with()


def test_cloud_key_is_redacted_and_vm_has_lifetime_limit(vm):
  machine, create = vm
  executor = SmolCodeExecutor(target='cloud', api_key='private-token')

  assert 'private-token' not in repr(executor)
  assert 'private-token' not in executor.model_dump_json()
  executor.execute_code(None, CodeExecutionInput(code='print(42)'))

  config, conn = create.call_args.args
  assert conn.target == 'cloud'
  assert conn.api_key == 'private-token'
  assert config.ttl_seconds >= 2 * executor.timeout_seconds
  assert config.auto_stop_seconds > executor.timeout_seconds
  machine.delete.assert_called_once_with()


def test_input_files_are_uploaded_into_the_sandbox(vm):
  machine, _ = vm
  executor = SmolCodeExecutor(optimize_data_file=True)
  executor.execute_code(
      None,
      CodeExecutionInput(
          code="print(open('data.csv').read())",
          input_files=[
              File(name='data.csv', content='one,two', mime_type='text/csv'),
              File(name='file.bin', content=b'\x00\xff'),
          ],
      ),
  )

  assert machine.write_file.call_args_list[0].args == (
      '/workspace/data.csv',
      b'one,two',
  )
  assert machine.write_file.call_args_list[1].args == (
      '/workspace/file.bin',
      b'\x00\xff',
  )
  assert executor.optimize_data_file is True
  machine.delete.assert_called_once_with()


@pytest.mark.parametrize('name', ['../secret', '/etc/passwd', 'a\\b', '.', ''])
def test_invalid_file_path_is_rejected_before_provisioning(vm, name):
  _, create = vm
  with pytest.raises(ValueError, match='Invalid sandbox input file name'):
    SmolCodeExecutor().execute_code(
        None,
        CodeExecutionInput(
            code='pass', input_files=[File(name=name, content='')]
        ),
    )
  create.assert_not_called()


def test_exec_failure_deletes_vm(vm):
  machine, _ = vm
  machine.exec.side_effect = RuntimeError('timed out')
  with pytest.raises(RuntimeError, match='timed out'):
    SmolCodeExecutor(timeout_seconds=1).execute_code(
        None, CodeExecutionInput(code='while True: pass')
    )
  machine.delete.assert_called_once_with()


def test_delete_failure_does_not_hide_execution_failure(vm, caplog):
  machine, _ = vm
  machine.exec.side_effect = RuntimeError('execution failed')
  machine.delete.side_effect = RuntimeError('delete failed')
  with pytest.raises(RuntimeError, match='execution failed'):
    SmolCodeExecutor().execute_code(None, CodeExecutionInput(code='pass'))
  assert 'Could not delete Smol VM' in caplog.text


def test_delete_failure_is_reported_after_success(vm):
  machine, _ = vm
  machine.delete.side_effect = RuntimeError('delete failed')
  with pytest.raises(RuntimeError, match='delete failed'):
    SmolCodeExecutor().execute_code(None, CodeExecutionInput(code='pass'))


def test_truncated_output_is_visible(vm):
  machine, _ = vm
  machine.exec.return_value.stdout_truncated = True
  result = SmolCodeExecutor().execute_code(
      None, CodeExecutionInput(code='print(42)')
  )
  assert 'truncated' in result.stderr


@pytest.mark.parametrize(
    'options', [{'stateful': True}, {'timeout_seconds': 0}]
)
def test_unbounded_or_stateful_config_is_rejected(options):
  with pytest.raises(ValueError):
    SmolCodeExecutor(**options)


def test_missing_optional_sdk_gives_install_hint(monkeypatch):
  monkeypatch.setitem(sys.modules, 'smol', None)
  with pytest.raises(ImportError, match=r'google-adk\[smol\]'):
    SmolCodeExecutor().execute_code(None, CodeExecutionInput(code='pass'))
