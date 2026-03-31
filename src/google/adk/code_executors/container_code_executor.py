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
import logging
import os
import threading
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

logger = logging.getLogger('google_adk.' + __name__)
DEFAULT_IMAGE_TAG = 'adk-code-executor:latest'


class ContainerCodeExecutor(BaseCodeExecutor):
  """A code executor that uses a custom container to execute code.

  Attributes:
    base_url: Optional. The base url of the user hosted Docker client.
    image: The tag of the predefined image or custom image to run on the
      container. Either docker_path or image must be set.
    docker_path: The path to the directory containing the Dockerfile. If set,
      build the image from the dockerfile path instead of using the predefined
      image. Either docker_path or image must be set.
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
  predefined image. Either docker_path or image must be set.
  """

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  _client: DockerClient = None
  _container: Container = None

  def __init__(
      self,
      base_url: Optional[str] = None,
      image: Optional[str] = None,
      docker_path: Optional[str] = None,
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
      **data: The data to initialize the ContainerCodeExecutor.
    """
    if not image and not docker_path:
      raise ValueError(
          'Either image or docker_path must be set for ContainerCodeExecutor.'
      )
    if 'stateful' in data and data['stateful']:
      raise ValueError('Cannot set `stateful=True` in ContainerCodeExecutor.')
    if 'optimize_data_file' in data and data['optimize_data_file']:
      raise ValueError(
          'Cannot set `optimize_data_file=True` in ContainerCodeExecutor.'
      )

    super().__init__(**data)
    self.base_url = base_url
    self.image = image if image else DEFAULT_IMAGE_TAG
    self.docker_path = os.path.abspath(docker_path) if docker_path else None

    self._client = (
        docker.from_env()
        if not self.base_url
        else docker.DockerClient(base_url=self.base_url)
    )
    # Initialize the container.
    self.__init_container()

    # Close the container when the on exit.
    atexit.register(self.__cleanup_container)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    logger.debug('Executing code:\n```\n%s\n```', code_execution_input.code)

    timeout = self.timeout_seconds
    exec_id = self._client.api.exec_create(
        self._container.id,
        ['python3', '-c', code_execution_input.code],
    )['Id']

    # exec_start blocks until the command finishes — run in a daemon
    # thread so we can enforce timeout.
    result_holder = {}

    def _run():
      try:
        result_holder['data'] = self._client.api.exec_start(exec_id, demux=True)
      except Exception as e:
        result_holder['error'] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    if timeout is not None:
      t.join(timeout=timeout)
    else:
      t.join()

    if t.is_alive():
      # Timeout: best-effort kill of the exec process inside the
      # container. NOTE: the blocked daemon thread may linger on the
      # Docker socket until the container is cleaned up. Repeated
      # timeouts can accumulate such threads. This is acceptable as
      # a first-pass fix; a stronger approach would restart the
      # container on timeout.
      try:
        inspect = self._client.api.exec_inspect(exec_id)
        pid = inspect.get('Pid')
        if pid:
          self._container.exec_run(['kill', '-9', str(pid)])
      except Exception:
        pass
      return CodeExecutionResult(
          stdout='',
          stderr=f'Code execution timed out after {timeout} seconds.',
          output_files=[],
      )

    # Propagate exceptions from exec_start (e.g. Docker API errors).
    if 'error' in result_holder:
      raise result_holder['error']

    data = result_holder.get('data', (None, None))
    output = data[0].decode('utf-8') if data and data[0] else ''
    error = (
        data[1].decode('utf-8') if data and len(data) > 1 and data[1] else ''
    )

    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )

  def _build_docker_image(self):
    """Builds the Docker image."""
    if not self.docker_path:
      raise ValueError('Docker path is not set.')
    if not os.path.exists(self.docker_path):
      raise FileNotFoundError(f'Invalid Docker path: {self.docker_path}')

    logger.info('Building Docker image...')
    self._client.images.build(
        path=self.docker_path,
        tag=self.image,
        rm=True,
    )
    logger.info('Docker image: %s built.', self.image)

  def _verify_python_installation(self):
    """Verifies the container has python3 installed."""
    exec_result = self._container.exec_run(['which', 'python3'])
    if exec_result.exit_code != 0:
      raise ValueError('python3 is not installed in the container.')

  def __init_container(self):
    """Initializes the container."""
    if not self._client:
      raise RuntimeError('Docker client is not initialized.')

    if self.docker_path:
      self._build_docker_image()

    logger.info('Starting container for ContainerCodeExecutor...')
    self._container = self._client.containers.run(
        image=self.image,
        detach=True,
        tty=True,
    )
    logger.info('Container %s started.', self._container.id)

    # Verify the container is able to run python3.
    self._verify_python_installation()

  def __cleanup_container(self):
    """Closes the container on exit."""
    if not self._container:
      return

    logger.info('[Cleanup] Stopping the container...')
    self._container.stop()
    self._container.remove()
    logger.info('Container %s stopped and removed.', self._container.id)
