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

from google.adk.code_executors.code_execution_utils import File
from google.adk.code_executors.code_executor_context import CodeExecutorContext
from google.adk.sessions.state import State
import pytest


@pytest.fixture
def empty_state() -> State:
  """Fixture for an empty session state."""
  return State({}, {})


@pytest.fixture
def context_with_data() -> CodeExecutorContext:
  """Fixture for a CodeExecutorContext with some prepopulated data."""
  state_data = {
      "_code_execution_context": {
          "execution_session_id": "session123",
          "processed_input_files": ["file1.csv", "file2.txt"],
      },
      "_code_executor_input_files": [
          {"name": "input1.txt", "content": "YQ==", "mime_type": "text/plain"}
      ],
      "_code_executor_error_counts": {"invocationA": 2},
  }
  state = State(state_data, {})
  return CodeExecutorContext(state)


def test_init_empty_state(empty_state: State):
  """Test initialization with an empty state."""
  ctx = CodeExecutorContext(empty_state)
  assert ctx._context == {}
  assert ctx._session_state is empty_state


def test_get_state_delta_empty(empty_state: State):
  """Test get_state_delta when context is empty."""
  ctx = CodeExecutorContext(empty_state)
  delta = ctx.get_state_delta()
  assert delta == {"_code_execution_context": {}}


def test_get_state_delta_with_data(context_with_data: CodeExecutorContext):
  """Test get_state_delta with existing context data."""
  delta = context_with_data.get_state_delta()
  expected_context = {
      "execution_session_id": "session123",
      "processed_input_files": ["file1.csv", "file2.txt"],
  }
  assert delta == {"_code_execution_context": expected_context}


def test_get_execution_id_exists(context_with_data: CodeExecutorContext):
  """Test getting an existing execution ID."""
  assert context_with_data.get_execution_id() == "session123"


def test_get_execution_id_not_exists(empty_state: State):
  """Test getting execution ID when it doesn't exist."""
  ctx = CodeExecutorContext(empty_state)
  assert ctx.get_execution_id() is None


def test_set_execution_id(empty_state: State):
  """Test setting an execution ID."""
  ctx = CodeExecutorContext(empty_state)
  ctx.set_execution_id("new_session_id")
  assert ctx._context["execution_session_id"] == "new_session_id"
  assert ctx.get_execution_id() == "new_session_id"


def test_get_processed_file_names_exists(
    context_with_data: CodeExecutorContext,
):
  """Test getting existing processed file names."""
  assert context_with_data.get_processed_file_names() == [
      "file1.csv",
      "file2.txt",
  ]


def test_get_processed_file_names_not_exists(empty_state: State):
  """Test getting processed file names when none exist."""
  ctx = CodeExecutorContext(empty_state)
  assert ctx.get_processed_file_names() == []


def test_add_processed_file_names_new(empty_state: State):
  """Test adding processed file names to an empty context."""
  ctx = CodeExecutorContext(empty_state)
  ctx.add_processed_file_names(["new_file.py"])
  assert ctx._context["processed_input_files"] == ["new_file.py"]


def test_add_processed_file_names_append(
    context_with_data: CodeExecutorContext,
):
  """Test appending to existing processed file names."""
  context_with_data.add_processed_file_names(["another_file.md"])
  assert context_with_data.get_processed_file_names() == [
      "file1.csv",
      "file2.txt",
      "another_file.md",
  ]


def test_get_input_files_exists(context_with_data: CodeExecutorContext):
  """Test getting existing input files."""
  files = context_with_data.get_input_files()
  assert len(files) == 1
  assert files[0].name == "input1.txt"
  assert files[0].content == "YQ=="
  assert files[0].mime_type == "text/plain"


def test_get_input_files_not_exists(empty_state: State):
  """Test getting input files when none exist."""
  ctx = CodeExecutorContext(empty_state)
  assert ctx.get_input_files() == []


def test_add_input_files_new(empty_state: State):
  """Test adding input files to an empty session state."""
  ctx = CodeExecutorContext(empty_state)
  new_files = [
      File(name="new.dat", content="Yg==", mime_type="application/octet-stream")
  ]
  ctx.add_input_files(new_files)
  assert empty_state["_code_executor_input_files"] == [{
      "name": "new.dat",
      "content": "Yg==",
      "mime_type": "application/octet-stream",
  }]


def test_add_input_files_append(context_with_data: CodeExecutorContext):
  """Test appending to existing input files."""
  new_file = File(name="input2.log", content="Yw==", mime_type="text/x-log")
  context_with_data.add_input_files([new_file])
  expected_files_data = [
      {"name": "input1.txt", "content": "YQ==", "mime_type": "text/plain"},
      {"name": "input2.log", "content": "Yw==", "mime_type": "text/x-log"},
  ]
  assert (
      context_with_data._session_state["_code_executor_input_files"]
      == expected_files_data
  )


def test_clear_input_files(context_with_data: CodeExecutorContext):
  """Test clearing input files and processed file names."""
  context_with_data.clear_input_files()
  assert context_with_data._session_state["_code_executor_input_files"] == []
  assert context_with_data._context["processed_input_files"] == []


def test_clear_input_files_when_not_exist(empty_state: State):
  """Test clearing input files when they don't exist initially."""
  ctx = CodeExecutorContext(empty_state)
  ctx.clear_input_files()  # Should not raise error
  assert "_code_executor_input_files" not in empty_state  # Or assert it's empty
  assert "_code_execution_context" not in empty_state or not empty_state[
      "_code_execution_context"
  ].get("processed_input_files")


def test_get_error_count_exists(context_with_data: CodeExecutorContext):
  """Test getting an existing error count."""
  assert context_with_data.get_error_count("invocationA") == 2


def test_get_error_count_invocation_not_exists(
    context_with_data: CodeExecutorContext,
):
  """Test getting error count for an unknown invocation ID."""
  assert context_with_data.get_error_count("invocationB") == 0


def test_get_error_count_no_error_key(empty_state: State):
  """Test getting error count when the error key itself doesn't exist."""
  ctx = CodeExecutorContext(empty_state)
  assert ctx.get_error_count("any_invocation") == 0


def test_increment_error_count_new_invocation(empty_state: State):
  """Test incrementing error count for a new invocation ID."""
  ctx = CodeExecutorContext(empty_state)
  ctx.increment_error_count("invocationNew")
  assert empty_state["_code_executor_error_counts"]["invocationNew"] == 1


def test_increment_error_count_existing_invocation(
    context_with_data: CodeExecutorContext,
):
  """Test incrementing error count for an existing invocation ID."""
  context_with_data.increment_error_count("invocationA")
  assert (
      context_with_data._session_state["_code_executor_error_counts"][
          "invocationA"
      ]
      == 3
  )


def test_reset_error_count_exists(context_with_data: CodeExecutorContext):
  """Test resetting an existing error count."""
  context_with_data.reset_error_count("invocationA")
  assert "invocationA" not in (
      context_with_data._session_state["_code_executor_error_counts"]
  )


def test_reset_error_count_not_exists(context_with_data: CodeExecutorContext):
  """Test resetting an error count that doesn't exist."""
  context_with_data.reset_error_count("invocationB")  # Should not raise
  assert "invocationB" not in (
      context_with_data._session_state["_code_executor_error_counts"]
  )


def test_reset_error_count_no_error_key(empty_state: State):
  """Test resetting when the error key itself doesn't exist."""
  ctx = CodeExecutorContext(empty_state)
  ctx.reset_error_count("any_invocation")  # Should not raise
  assert "_code_executor_error_counts" not in empty_state


def test_update_code_execution_result_new_invocation(empty_state: State):
  """Test updating code execution result for a new invocation."""
  ctx = CodeExecutorContext(empty_state)
  ctx.update_code_execution_result("inv1", "print('hi')", "hi", "")
  results = empty_state["_code_execution_results"]["inv1"]
  assert len(results) == 1
  assert results[0]["code"] == "print('hi')"
  assert results[0]["result_stdout"] == "hi"
  assert results[0]["result_stderr"] == ""
  assert "timestamp" in results[0]


def test_update_code_execution_result_append(
    context_with_data: CodeExecutorContext,
):
  """Test appending to existing code execution results for an invocation."""
  # First, let's add an initial result for a new invocation to the existing state
  context_with_data._session_state["_code_execution_results"] = {
      "invocationX": [{
          "code": "old_code",
          "result_stdout": "old_out",
          "result_stderr": "old_err",
          "timestamp": 123,
      }]
  }
  context_with_data.update_code_execution_result(
      "invocationX", "new_code", "new_out", "new_err"
  )
  results = context_with_data._session_state["_code_execution_results"][
      "invocationX"
  ]
  assert len(results) == 2
  assert results[1]["code"] == "new_code"
  assert results[1]["result_stdout"] == "new_out"
  assert results[1]["result_stderr"] == "new_err"
