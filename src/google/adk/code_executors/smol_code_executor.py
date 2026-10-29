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

"""Run model-generated Python in a disposable local or Cloud microVM."""

from __future__ import annotations

import logging
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING
import uuid

from pydantic import Field
from pydantic import SecretStr
from typing_extensions import override

from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class SmolCodeExecutor(BaseCodeExecutor):
  """Execute Python in a fresh Smol Machines microVM for each code block.

  A local VM needs Linux KVM or macOS Hypervisor.framework. Use ``target='cloud'``
  for Smol Cloud, with ``SMOL_CLOUD_TOKEN`` or a configured Smol CLI login.
  Install the optional SDK with ``pip install 'google-adk[smol]'``.

  Guest network access is disabled by default. The VM is deleted after every
  execution, including failed or timed-out executions; no Python variables or
  files persist between blocks. Input files are available in ``/workspace``.
  """

  target: Literal['local', 'cloud'] = 'local'
  image: str = 'python:3.12-slim'
  network_enabled: bool = False
  cpus: int = Field(default=2, gt=0)
  memory_mb: int = Field(default=1024, gt=0)
  api_key: SecretStr | None = Field(default=None, repr=False)
  timeout_seconds: int = Field(default=300, gt=0)
  stateful: bool = Field(default=False, frozen=True)
  optimize_data_file: bool = False

  def __init__(self, **data: Any) -> None:
    if data.get('stateful'):
      raise ValueError('SmolCodeExecutor creates a new VM for each execution.')
    super().__init__(**data)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    del invocation_context  # A fresh VM has no session state to share.
    for file in code_execution_input.input_files:
      if (
          not file.name
          or file.name in ('.', '..')
          or '/' in file.name
          or '\\' in file.name
          or '\x00' in file.name
      ):
        raise ValueError(f'Invalid sandbox input file name: {file.name!r}')
    try:
      import smol
    except ImportError as exc:
      raise ImportError(
          'SmolCodeExecutor requires the Smol Machines SDK. '
          "Install with: pip install 'google-adk[smol]'"
      ) from exc

    connection = smol.ConnectOptions(
        target=self.target,
        api_key=self.api_key.get_secret_value() if self.api_key else None,
    )
    config = smol.MachineConfig(
        image=self.image,
        resources=smol.ResourceSpec(
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            network=self.network_enabled,
        ),
        persistent=False,
        auto_stop_seconds=(
            max(600, self.timeout_seconds + 60)
            if self.target == 'cloud'
            else None
        ),
        ttl_seconds=(
            max(1200, 2 * self.timeout_seconds + 60)
            if self.target == 'cloud'
            else None
        ),
    )
    machine = smol.Machine.create(config, connection)
    failed = False
    try:
      script_path = f'/workspace/adk-code-{uuid.uuid4().hex}.py'
      for file in code_execution_input.input_files:
        content = (
            file.content.encode('utf-8')
            if isinstance(file.content, str)
            else file.content
        )
        machine.write_file(f'/workspace/{file.name}', content)
      machine.write_file(script_path, code_execution_input.code)
      result = machine.exec(
          ['python3', script_path],
          smol.ExecOptions(
              workdir='/workspace', timeout=float(self.timeout_seconds)
          ),
      )
      stderr = result.stderr
      if result.stdout_truncated or result.stderr_truncated:
        stderr += '\nSmol VM output was truncated.'
      return CodeExecutionResult(
          stdout=result.stdout,
          stderr=stderr,
          exit_code=result.exit_code,
      )
    except BaseException:
      failed = True
      raise
    finally:
      try:
        machine.delete()
      except Exception:
        if failed:
          logger.exception('Could not delete Smol VM after execution failure')
        else:
          raise
