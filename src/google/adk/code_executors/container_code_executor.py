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

import atexit
import io
import logging
import os
import tarfile
from typing import Optional

import docker
from docker.client import DockerClient
from docker.models.containers import Container
from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult
from .code_execution_utils import File

logger = logging.getLogger("google_adk." + __name__)
DEFAULT_IMAGE_TAG = "adk-code-executor:latest"


class ContainerCodeExecutor(BaseCodeExecutor):
  """A code executor that uses a custom container to execute code.

  Attributes:
    base_url: Optional. The base url of the user hosted Docker client.
    image: The tag of the predefined image or custom image to run on the
      container. Either docker_path or image must be set.
    docker_path: The path to the directory containing the Dockerfile. If set,
      build the image from the dockerfile path instead of using the predefined
      image. Either docker_path or image must be set.
    input_dir: The directory in the container where input files will be placed.
    output_dir: The directory in the container where output files will be
      retrieved from.
  """

  base_url: Optional[str] = None
  """
  Optional. The base url of the user hosted Docker client.
  """

  image: str = None
  """
  The tag of the predefined image or custom image to run on the container.
  Either docker_path or image must be set.
  """

  docker_path: str = None
  """
  The path to the directory containing the Dockerfile.
  If set, build the image from the dockerfile path instead of using the
  predefined image. Either docker_path or image must be set .
  """

  input_dir: str = "/tmp/inputs"
  """
  The directory in the container where input files will be placed.
  """

  output_dir: str = "/tmp/outputs"
  """
  The directory in the container where output files will be retrieved from.
  """

  stateful: bool = Field(default=False, frozen=True, exclude=True)

  optimize_data_file: bool = Field(default=True, frozen=True, exclude=True)

  _client: DockerClient = None
  _container: Container = None

  def __init__(
      self,
      base_url: Optional[str] = None,
      image: Optional[str] = None,
      docker_path: Optional[str] = None,
      input_dir: Optional[str] = None,
      output_dir: Optional[str] = None,
      **data,
  ):
    """Initializes the ContainerCodeExecutor.

    Args:
      base_url: Optional. The base url of the user hosted Docker client.
      image: The tag of the predefined image or custom image to run on the
        container. Either docker_path or image must be set.
      docker_path: The path to the directory containing the Dockerfile. If set,
        build the image from the dockerfile path instead of using the predefined
        image. Either docker_path or image must be set.
      input_dir: The directory in the container where input files will be placed.
        Defaults to '/tmp/inputs'.
      output_dir: The directory in the container where output files will be
        retrieved from. Defaults to '/tmp/outputs'.
      **data: The data to initialize the ContainerCodeExecutor.
    """
    if not image and not docker_path:
      raise ValueError(
          "Either image or docker_path must be set for ContainerCodeExecutor."
      )
    if "stateful" in data and data["stateful"]:
      raise ValueError("Cannot set `stateful=True` in ContainerCodeExecutor.")

    super().__init__(**data)
    self.base_url = base_url
    self.image = image if image else DEFAULT_IMAGE_TAG
    self.docker_path = os.path.abspath(docker_path) if docker_path else None
    self.input_dir = input_dir if input_dir else "/tmp/inputs"
    self.output_dir = output_dir if output_dir else "/tmp/outputs"

    self._client = (
        docker.from_env()
        if not self.base_url
        else docker.DockerClient(base_url=self.base_url)
    )
    self.__init_container()

    atexit.register(self.__cleanup_container)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    if code_execution_input.input_files:
      self._put_input_files(code_execution_input.input_files)

    self._create_output_directory()

    output = ""
    error = ""
    exec_result = self._container.exec_run(
        ["python3", "-c", code_execution_input.code],
        demux=True,
    )
    logger.debug("Executed code:\n```\n%s\n```", code_execution_input.code)

    if exec_result.output and exec_result.output[0]:
      output = exec_result.output[0].decode("utf-8")
    if (
        exec_result.output
        and len(exec_result.output) > 1
        and exec_result.output[1]
    ):
      error = exec_result.output[1].decode("utf-8")

    output_files = self._get_output_files()

    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=output_files,
    )

  def _build_docker_image(self):
    """Builds the Docker image."""
    if not self.docker_path:
      raise ValueError("Docker path is not set.")
    if not os.path.exists(self.docker_path):
      raise FileNotFoundError(f"Invalid Docker path: {self.docker_path}")

    logger.info("Building Docker image...")
    self._client.images.build(
        path=self.docker_path,
        tag=self.image,
        rm=True,
    )
    logger.info("Docker image: %s built.", self.image)

  def _verify_python_installation(self):
    """Verifies the container has python3 installed."""
    exec_result = self._container.exec_run(["which", "python3"])
    if exec_result.exit_code != 0:
      raise ValueError("python3 is not installed in the container.")

  def _put_input_files(self, input_files: list[File]) -> None:
    """Puts input files into the container using put_archive.

    Args:
      input_files: The list of input files to copy into the container.
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
      for file in input_files:
        content = file.content
        if isinstance(content, str):
          content = content.encode("utf-8")
        tarinfo = tarfile.TarInfo(name=file.name)
        tarinfo.size = len(content)
        tar.addfile(tarinfo, io.BytesIO(content))

    tar_buffer.seek(0)
    self._container.put_archive(
        self.input_dir,
        tar_buffer.read(),
    )
    logger.debug(
        "Copied %d input files to %s", len(input_files), self.input_dir
    )

  def _create_output_directory(self) -> None:
    """Creates the output directory in the container if it doesn't exist."""
    exec_result = self._container.exec_run(
        ["mkdir", "-p", self.output_dir],
    )
    if exec_result.exit_code != 0:
      logger.warning(
          "Failed to create output directory %s: %s",
          self.output_dir,
          exec_result.output,
      )

  def _get_output_files(self) -> list[File]:
    """Gets output files from the container.

    Returns:
      The list of output files retrieved from the container.
    """
    try:
      tar_stream, _ = self._container.get_archive(self.output_dir)
    except docker.errors.APIError as e:
      if e.response.status_code == 404:
        logger.debug("No output files found at %s", self.output_dir)
        return []
      raise
    if isinstance(tar_stream, bytes):
      tar_bytes = tar_stream
    else:
      tar_bytes = b"".join(tar_stream)
    tar_buffer = io.BytesIO(tar_bytes)
    output_files = []

    with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
      for member in tar.getmembers():
        if member.isfile():
          file_obj = tar.extractfile(member)
          if file_obj:
            content = file_obj.read()
            file_name = os.path.basename(member.name)
            if file_name:
              output_files.append(
                  File(
                      name=file_name,
                      content=content,
                      mime_type=self._guess_mime_type(file_name),
                  )
              )

    logger.debug(
        "Retrieved %d output files from %s", len(output_files), self.output_dir
    )
    return output_files

  def _guess_mime_type(self, file_name: str) -> str:
    """Guesses the MIME type based on the file extension.

    Args:
      file_name: The name of the file.

    Returns:
      The guessed MIME type, or 'application/octet-stream' if unknown.
    """
    import mimetypes

    mime_type, _ = mimetypes.guess_type(file_name)
    return mime_type if mime_type else "application/octet-stream"

  def __init_container(self):
    """Initializes the container."""
    if not self._client:
      raise RuntimeError("Docker client is not initialized.")

    if self.docker_path:
      self._build_docker_image()

    logger.info("Starting container for ContainerCodeExecutor...")
    self._container = self._client.containers.run(
        image=self.image,
        detach=True,
        tty=True,
    )
    logger.info("Container %s started.", self._container.id)

    self._verify_python_installation()

  def __cleanup_container(self):
    """Closes the container on exit."""
    if not self._container:
      return

    logger.info("[Cleanup] Stopping the container...")
    self._container.stop()
    self._container.remove()
    logger.info("Container %s stopped and removed.", self._container.id)
