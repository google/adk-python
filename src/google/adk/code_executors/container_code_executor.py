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
from typing import Any
from typing import cast

import docker
from docker.client import DockerClient
from docker.models.containers import Container
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)
DEFAULT_IMAGE_TAG = 'adk-code-executor:latest'

# Reported by the supervisor below when it kills a run that hit the bound.
# Follows the convention of coreutils `timeout`; unlike 128 + SIGALRM it is not
# something the executed code produces by letting an alarm of its own fire.
_TIMEOUT_EXIT_CODE = 124

# Runs the code under a supervisor that enforces a hard wall-clock bound inside
# the container. The code runs in a forked child in its own process group; the
# supervisor waits for it and, when the bound expires, SIGKILLs the whole group
# and exits with `_TIMEOUT_EXIT_CODE`. The group is also swept once the code
# finishes normally, so nothing it started is left running in the shared
# container (a leftover process would also hold the exec's output open).
#
# Two properties matter for code that may be hostile: the deadline lives in a
# process the code never runs in, so it cannot be disarmed from inside (an
# alarm armed in the executing process could be cancelled with one call), and
# SIGKILL to the group reaches what the code spawned, not just its top frame.
# The code is passed as an argument rather than inlined so that tracebacks keep
# the original line numbers, and argv is restored to what `python3 -c` would
# have given so that code parsing arguments still works.
#
# Not covered: code that deliberately leaves the group (`os.setsid()`) or
# double-forks away survives the bound until the container is torn down.
_TIMEOUT_WRAPPER = """\
import os, signal, sys

_timeout = int(sys.argv[1])
_source = sys.argv[2]
del sys.argv[1:]

_pid = os.fork()
if _pid == 0:
  try:
    os.setpgid(0, 0)
  except OSError:
    pass
  exec(compile(_source, '<adk_code>', 'exec'), {'__name__': '__main__'})
else:

  def _sweep_group():
    try:
      os.killpg(_pid, signal.SIGKILL)
    except OSError:
      pass

  def _expire(_signum, _frame):
    _sweep_group()
    try:
      os.kill(_pid, signal.SIGKILL)
    except OSError:
      pass
    os._exit(%d)

  try:
    os.setpgid(_pid, _pid)
  except OSError:
    pass
  signal.signal(signal.SIGALRM, _expire)
  signal.alarm(_timeout)
  _status = os.waitpid(_pid, 0)[1]
  signal.alarm(0)
  _sweep_group()
  os._exit(
      128 + os.WTERMSIG(_status)
      if os.WIFSIGNALED(_status)
      else os.WEXITSTATUS(_status)
  )
""" % _TIMEOUT_EXIT_CODE


class ContainerCodeExecutor(BaseCodeExecutor):
  """A code executor that uses a custom container to execute code.

  Security note: this executor runs model-generated code, which may be
  influenced by untrusted input (e.g. via prompt injection). By default the
  container is started with networking disabled and all Linux capabilities
  dropped so that the executed code cannot reach the network (including the
  cloud metadata endpoint at ``169.254.169.254``) or escalate privileges. For
  stronger, kernel-level isolation of untrusted code prefer
  ``GkeCodeExecutor`` (gVisor) or a managed executor
  (``VertexAiCodeExecutor`` / ``AgentEngineSandboxCodeExecutor``).

  Attributes:
    base_url: Optional. The base url of the user hosted Docker client.
    image: The tag of the predefined image or custom image to run on the
      container. Either docker_path or image must be set.
    docker_path: The path to the directory containing the Dockerfile. If set,
      build the image from the dockerfile path instead of using the predefined
      image. Either docker_path or image must be set.
    network_enabled: Whether to start the container with networking enabled.
      Defaults to False. Set to True only if the executed code must make network
      requests and you trust it.
  """

  base_url: str | None = None
  """
  Optional. The base url of the user hosted Docker client.
  """

  image: str = DEFAULT_IMAGE_TAG
  """
  The tag of the predefined image or custom image to run on the container.
  Either docker_path or image must be set.
  """

  docker_path: str | None = None
  """
  The path to the directory containing the Dockerfile.
  If set, build the image from the dockerfile path instead of using the
  predefined image. Either docker_path or image must be set.
  """

  network_enabled: bool = False
  """
  Whether to start the code execution container with networking enabled.

  Defaults to False so that untrusted, model-generated code cannot reach the
  network -- in particular the cloud metadata endpoint at 169.254.169.254
  (which can yield the host's service-account credentials), internal services,
  or arbitrary exfiltration destinations. Set to True only if the executed
  code must make network requests and you trust it.
  """

  # Overrides the BaseCodeExecutor attribute: unlike the base default of None,
  # the timeout here is always finite and must be positive (0 would mean no
  # bound at all).
  timeout_seconds: int = Field(default=300, gt=0)
  """The wall-clock timeout in seconds for a single code execution.

  Every execution shares one long-lived container, so an unbounded run (e.g. a
  loop emitted by the model) would keep burning that container's CPU for every
  later caller. Defaults to 300, matching ``GkeCodeExecutor``. A computation
  that legitimately runs longer than the timeout is killed, so raise it rather
  than removing it; ``None`` is rejected, unlike on the base class.

  When the timeout expires the executed code is killed along with the process
  group it runs in, so what it spawned goes with it. Code that deliberately
  detaches from that group keeps running until the container is torn down.
  """

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  _client: DockerClient | None = PrivateAttr(default=None)
  _container: Container | None = PrivateAttr(default=None)

  def __init__(
      self,
      base_url: str | None = None,
      image: str | None = None,
      docker_path: str | None = None,
      **data: Any,
  ) -> None:
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
    output = ''
    error = ''
    exec_result = cast(Container, self._container).exec_run(
        [
            'python3',
            '-c',
            _TIMEOUT_WRAPPER,
            str(self.timeout_seconds),
            code_execution_input.code,
        ],
        demux=True,
    )
    logger.debug('Executed code:\n```\n%s\n```', code_execution_input.code)

    if exec_result.output and exec_result.output[0]:
      output = exec_result.output[0].decode('utf-8')
    if (
        exec_result.output
        and len(exec_result.output) > 1
        and exec_result.output[1]
    ):
      error = exec_result.output[1].decode('utf-8')

    if exec_result.exit_code == _TIMEOUT_EXIT_CODE:
      # Appended rather than assigned: whatever the code managed to write to
      # stderr before the alarm fired is still the useful diagnostic, but on
      # its own it would hide the fact that the run was cut short.
      timed_out = (
          f'Code execution timed out after {self.timeout_seconds} seconds.'
      )
      error = f'{error}\n{timed_out}' if error else timed_out

    # Collect the final result.
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )

  def _build_docker_image(self) -> None:
    """Builds the Docker image."""
    if not self.docker_path:
      raise ValueError('Docker path is not set.')
    if not os.path.exists(self.docker_path):
      raise FileNotFoundError(f'Invalid Docker path: {self.docker_path}')

    logger.info('Building Docker image...')
    cast(DockerClient, self._client).images.build(
        path=self.docker_path,
        tag=self.image,
        rm=True,
    )
    logger.info('Docker image: %s built.', self.image)

  def _verify_python_installation(self) -> None:
    """Verifies the container has python3 installed."""
    exec_result = cast(Container, self._container).exec_run(
        ['which', 'python3']
    )
    if exec_result.exit_code != 0:
      raise ValueError('python3 is not installed in the container.')

  def __init_container(self) -> None:
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
        # Harden the sandbox for untrusted, model-generated code: no network
        # (blocks metadata/SSRF/exfil), drop all Linux capabilities, and
        # forbid privilege escalation. Networking can be re-enabled via
        # `network_enabled=True` when the executed code is trusted.
        network_disabled=not self.network_enabled,
        cap_drop=['ALL'],
        security_opt=['no-new-privileges'],
    )
    logger.info('Container %s started.', self._container.id)

    # Verify the container is able to run python3.
    self._verify_python_installation()

  def __cleanup_container(self) -> None:
    """Closes the container on exit."""
    if not self._container:
      return

    logger.info('[Cleanup] Stopping the container...')
    self._container.stop()
    self._container.remove()
    logger.info('Container %s stopped and removed.', self._container.id)
