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

"""Unit tests for ContainerCodeExecutor."""

import io
import tarfile
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.code_execution_utils import File
from google.adk.code_executors.container_code_executor import ContainerCodeExecutor
import pytest


@pytest.fixture
def mock_container():
  container = mock.MagicMock()
  container.id = "test-container-id"
  container.exec_run.return_value = mock.MagicMock(
      output=(b"Hello World", b""),
      exit_code=0,
  )
  container.get_archive.return_value = (b"", mock.MagicMock())
  return container


@pytest.fixture
def mock_docker_client(mock_container):
  client = mock.MagicMock()
  client.containers.run.return_value = mock_container
  return client


class TestContainerCodeExecutorInit:

  def test_init_with_image(self, mock_docker_client):
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")
      assert executor.image == "test-image:latest"
      assert executor.input_dir == "/tmp/inputs"
      assert executor.output_dir == "/tmp/outputs"

  def test_init_with_custom_dirs(self, mock_docker_client):
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(
          image="test-image:latest",
          input_dir="/custom/inputs",
          output_dir="/custom/outputs",
      )
      assert executor.input_dir == "/custom/inputs"
      assert executor.output_dir == "/custom/outputs"

  def test_init_requires_image_or_docker_path(self):
    with pytest.raises(
        ValueError, match="Either image or docker_path must be set"
    ):
      ContainerCodeExecutor()

  def test_init_rejects_stateful(self):
    with pytest.raises(ValueError, match="Cannot set `stateful=True`"):
      ContainerCodeExecutor(image="test", stateful=True)

  def test_init_allows_optimize_data_file(self, mock_docker_client):
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(
          image="test-image:latest", optimize_data_file=True
      )
      assert executor.optimize_data_file is True


class TestExecuteCode:

  def test_execute_code_basic(self, mock_docker_client, mock_container):
    import docker

    mock_container.get_archive.side_effect = docker.errors.APIError(
        "Not found", response=mock.MagicMock(status_code=404)
    )
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")
      context = mock.MagicMock(spec=InvocationContext)
      code_input = CodeExecutionInput(code='print("Hello World")')

      result = executor.execute_code(context, code_input)

      assert result.stdout == "Hello World"
      assert result.stderr == ""
      assert result.output_files == []

  def test_execute_code_with_error(self, mock_docker_client, mock_container):
    import docker

    call_count = [0]

    def exec_run_side_effect(cmd, demux=False):
      call_count[0] += 1
      if call_count[0] == 3:
        return mock.MagicMock(
            exit_code=1,
            output=(b"", b"Some error"),
        )
      return mock.MagicMock(exit_code=0, output=(b"", b""))

    mock_container.exec_run.side_effect = exec_run_side_effect
    mock_container.get_archive.side_effect = docker.errors.APIError(
        "Not found", response=mock.MagicMock(status_code=404)
    )
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")
      context = mock.MagicMock(spec=InvocationContext)
      code_input = CodeExecutionInput(code='raise Error("test")')

      result = executor.execute_code(context, code_input)

      assert result.stderr == "Some error"

  def test_execute_code_with_input_files(
      self, mock_docker_client, mock_container
  ):
    import docker

    mock_container.put_archive = mock.MagicMock()
    mock_container.exec_run.return_value = mock.MagicMock(
        output=(b"", b""),
        exit_code=0,
    )
    mock_container.get_archive.side_effect = docker.errors.APIError(
        "Not found", response=mock.MagicMock(status_code=404)
    )
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")
      context = mock.MagicMock(spec=InvocationContext)
      code_input = CodeExecutionInput(
          code='print("test")',
          input_files=[File(name="test.txt", content="test content")],
      )

      result = executor.execute_code(context, code_input)

      mock_container.put_archive.assert_called_once()
      call_args = mock_container.put_archive.call_args
      assert call_args[0][0] == "/tmp/inputs"

  def test_execute_code_with_output_files(
      self, mock_docker_client, mock_container
  ):
    mock_container.exec_run.return_value = mock.MagicMock(
        output=(b"", b""),
        exit_code=0,
    )

    content = b"output content"
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
      tar_info = tarfile.TarInfo(name="output.txt")
      tar_info.size = len(content)
      tar.addfile(tar_info, io.BytesIO(content))
    tar_buffer.seek(0)
    tar_bytes = tar_buffer.read()

    mock_container.get_archive.return_value = (tar_bytes, mock.MagicMock())

    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")
      context = mock.MagicMock(spec=InvocationContext)
      code_input = CodeExecutionInput(code='print("test")')

      result = executor.execute_code(context, code_input)

      assert len(result.output_files) == 1
      assert result.output_files[0].name == "output.txt"


class TestPutInputFiles:

  def test_put_archive_called(self, mock_docker_client, mock_container):
    mock_container.put_archive = mock.MagicMock()
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")
      input_files = [
          File(name="file1.txt", content="content1"),
          File(name="file2.txt", content="content2"),
      ]

      executor._put_input_files(input_files)

      mock_container.put_archive.assert_called_once()
      call_args = mock_container.put_archive.call_args
      assert call_args[0][0] == "/tmp/inputs"

  def test_handles_string_content(self, mock_docker_client, mock_container):
    mock_container.put_archive = mock.MagicMock()
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")
      input_files = [File(name="test.txt", content="string content")]

      executor._put_input_files(input_files)

      mock_container.put_archive.assert_called_once()


class TestGetOutputFiles:

  def test_no_output_files(self, mock_docker_client, mock_container):
    import docker

    mock_container.get_archive.side_effect = docker.errors.APIError(
        "Not found", response=mock.MagicMock(status_code=404)
    )
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")

      output_files = executor._get_output_files()

      assert output_files == []

  def test_extracts_files_from_archive(
      self, mock_docker_client, mock_container
  ):
    content = b"output content"
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
      tar_info = tarfile.TarInfo(name="output.txt")
      tar_info.size = len(content)
      tar.addfile(tar_info, io.BytesIO(content))
    tar_buffer.seek(0)
    tar_bytes = tar_buffer.read()

    mock_container.get_archive.return_value = (tar_bytes, mock.MagicMock())

    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")

      output_files = executor._get_output_files()

      assert len(output_files) == 1
      assert output_files[0].name == "output.txt"
      assert output_files[0].content == content


class TestMimeTypeGuessing:

  def test_guess_txt(self, mock_docker_client, mock_container):
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")

      mime_type = executor._guess_mime_type("test.txt")

      assert mime_type == "text/plain"

  def test_guess_csv(self, mock_docker_client, mock_container):
    import mimetypes as mimetypes_module

    original_guess_type = mimetypes_module.guess_type
    mimetypes_module.guess_type = lambda f: ("text/csv", None)
    try:
      with mock.patch("docker.from_env", return_value=mock_docker_client):
        executor = ContainerCodeExecutor(image="test-image:latest")

        mime_type = executor._guess_mime_type("data.csv")

        assert mime_type == "text/csv"
    finally:
      mimetypes_module.guess_type = original_guess_type

  def test_default_for_unknown(self, mock_docker_client, mock_container):
    with mock.patch("docker.from_env", return_value=mock_docker_client):
      executor = ContainerCodeExecutor(image="test-image:latest")

      mime_type = executor._guess_mime_type("unknown.xyz")

      assert mime_type == "application/octet-stream"
