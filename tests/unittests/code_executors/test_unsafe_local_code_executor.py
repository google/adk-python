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

import concurrent.futures
import os
import textwrap
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.code_execution_utils import File
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
import pytest


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Provides a mock InvocationContext."""
  mock_agent = MagicMock(spec=BaseAgent)
  mock_session = MagicMock(spec=Session)
  mock_session_service = MagicMock(spec=BaseSessionService)
  return InvocationContext(
      invocation_id="test_invocation",
      agent=mock_agent,
      session=mock_session,
      session_service=mock_session_service,
  )


class TestUnsafeLocalCodeExecutor:

  def test_init_default(self):
    executor = UnsafeLocalCodeExecutor()
    assert not executor.stateful
    assert not executor.optimize_data_file

  def test_init_stateful_raises_error(self):
    with pytest.raises(
        ValueError,
        match="Cannot set `stateful=True` in UnsafeLocalCodeExecutor.",
    ):
      UnsafeLocalCodeExecutor(stateful=True)

  def test_init_optimize_data_file_raises_error(self):
    with pytest.raises(
        ValueError,
        match=(
            "Cannot set `optimize_data_file=True` in UnsafeLocalCodeExecutor."
        ),
    ):
      UnsafeLocalCodeExecutor(optimize_data_file=True)

  def test_execute_code_simple_print(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == "hello world\n"
    assert result.stderr == ""
    assert result.output_files == []

  def test_execute_code_with_error(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='raise ValueError("Test error")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == ""
    assert "Test error" in result.stderr
    assert result.output_files == []

  def test_execute_code_variable_assignment(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="x = 10\nprint(x * 2)")
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "20\n"
    assert result.stderr == ""

  def test_execute_code_empty(self, mock_invocation_context: InvocationContext):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="")
    result = executor.execute_code(mock_invocation_context, code_input)
    assert result.stdout == ""
    assert result.stderr == ""

  def test_execute_code_nested_function_call(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code=(textwrap.dedent("""
                def helper(name):
                  return f'hi {name}'

                def run():
                  print(helper('ada'))

                run()
                """)))

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stderr == ""
    assert result.stdout == "hi ada\n"

  def test_concurrent_sandbox_and_plain_no_stdout_bleed(
      self, mock_invocation_context: InvocationContext
  ):
    """Concurrent sandbox and plain calls must not mix stdout."""
    executor = UnsafeLocalCodeExecutor()
    original_cwd = os.getcwd()
    plain_input = CodeExecutionInput(
        code='import os; print("PLAIN:" + os.getcwd())'
    )
    sandbox_input = CodeExecutionInput(
        code=(
            'import os, time; time.sleep(0.01); print("SANDBOX:" + os.getcwd())'
        ),
        input_files=[
            File(name="dummy.txt", content="data"),
        ],
        working_dir=".",
    )

    errors = []
    for _ in range(50):
      with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_sandbox = pool.submit(
            executor.execute_code,
            mock_invocation_context,
            sandbox_input,
        )
        f_plain = pool.submit(
            executor.execute_code,
            mock_invocation_context,
            plain_input,
        )
        r_sandbox = f_sandbox.result()
        r_plain = f_plain.result()

      if "PLAIN" in r_sandbox.stdout or "SANDBOX" in r_plain.stdout:
        errors.append(f"sandbox={r_sandbox.stdout!r} plain={r_plain.stdout!r}")

      # Plain-path cwd must remain the original cwd, not a temp dir
      plain_cwd = r_plain.stdout.strip().split(":", 1)[1]
      if plain_cwd != original_cwd:
        errors.append(f"plain cwd={plain_cwd!r} expected={original_cwd!r}")

    assert not errors, (
        f"bleed detected in {len(errors)}/50 iterations: " + errors[0]
    )
